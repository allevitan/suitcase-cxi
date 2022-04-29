# Suitcase subpackages should follow strict naming and interface conventions.
# The public API must include Serializer and should include export if it is
# intended to be user-facing. They should accept the parameters sketched here,
# but may also accpet additional required or optional keyword arguments, as
# needed.
import event_model
from pathlib import Path
import suitcase.utils
from ._version import get_versions
import h5py
import numpy as np
from datetime import datetime
from csxtools.fastccd import correct_images
from csxtools.image import rotate90


__version__ = get_versions()['version']
del get_versions


def export(gen, directory, file_prefix='{scan_id}_', **kwargs):
    """
    Export a stream of documents to cxi.

    .. note::

        This can alternatively be used to write data to generic buffers rather
        than creating files on disk. See the documentation for the
        ``directory`` parameter below.

    Parameters
    ----------
    gen : generator
        expected to yield ``(name, document)`` pairs

    directory : string, Path or Manager.
        For basic uses, this should be the path to the output directory given
        as a string or Path object. Use an empty string ``''`` to place files
        in the current working directory.

        In advanced applications, this may direct the serialized output to a
        memory buffer, network socket, or other writable buffer. It should be
        an instance of ``suitcase.utils.MemoryBufferManager`` and
        ``suitcase.utils.MultiFileManager`` or any object implementing that
        interface. See the suitcase documentation at
        https://nsls-ii.github.io/suitcase for details.

    file_prefix : str, optional
        The first part of the filename of the generated output files. This
        string may include templates as in ``{proposal_id}-{sample_name}-``,
        which are populated from the RunStart document. The default value is
        ``{uid}-`` which is guaranteed to be present and unique. A more
        descriptive value depends on the application and is therefore left to
        the user.

    **kwargs : kwargs
        Keyword arugments to be passed through to the underlying I/O library.

    Returns
    -------
    artifacts : dict
        dict mapping the 'labels' to lists of file names (or, in general,
        whatever resources are produced by the Manager)

    Examples
    --------

    Generate files with unique-identifier names in the current directory.

    >>> export(gen, '')

    Generate files with more readable metadata in the file names.

    >>> export(gen, '', '{plan_name}-{motors}-')

    Include the experiment's start time formatted as YYYY-MM-DD_HH-MM.

    >>> export(gen, '', '{time:%Y-%m-%d_%H:%M}-')

    Place the files in a different directory, such as on a mounted USB stick.

    >>> export(gen, '/path/to/my_usb_stick')
    """
    with Serializer(directory, file_prefix, **kwargs) as serializer:
        for item in gen:
            serializer(*item)

    return serializer.artifacts


class Serializer(event_model.DocumentRouter):
    """
    Serialize a stream of documents to cxi.

    .. note::

        This can alternatively be used to write data to generic buffers rather
        than creating files on disk. See the documentation for the
        ``directory`` parameter below.

    Parameters
    ----------
    directory : string, Path, or Manager
        For basic uses, this should be the path to the output directory given
        as a string or Path object. Use an empty string ``''`` to place files
        in the current working directory.

        In advanced applications, this may direct the serialized output to a
        memory buffer, network socket, or other writable buffer. It should be
        an instance of ``suitcase.utils.MemoryBufferManager`` and
        ``suitcase.utils.MultiFileManager`` or any object implementing that
        interface. See the suitcase documentation at
        https://nsls-ii.github.io/suitcase for details.

    file_prefix : str, optional
        The first part of the filename of the generated output files. This
        string may include templates as in ``{proposal_id}-{sample_name}-``,
        which are populated from the RunStart document. The default value is
        ``{uid}-`` which is guaranteed to be present and unique. A more
        descriptive value depends on the application and is therefore left to
        the user.

    **kwargs : kwargs
        Keyword arugments to be passed through to the underlying I/O library.

    Attributes
    ----------
    artifacts
        dict mapping the 'labels' to lists of file names (or, in general,
        whatever resources are produced by the Manager)
    """
    def __init__(self, directory, file_prefix='{scan_id}_', distance=0.3, pitch=[30e-6,30e-6], fccd_intersection=[440,431], dark=None, harmonic=1, mask_file=None, **kwargs):

        self._file_prefix = file_prefix
        self._distance = distance
        self._pitch = pitch
        self._harmonic = harmonic
        self._fccd_intersection = fccd_intersection # reasonable default
        self._kwargs = kwargs
        self._templated_file_prefix = ''  # set when we get a 'start' document
        
        if isinstance(directory, (str, Path)):
            # The user has given us a filepath; they want files.
            # Set up a MultiFileManager for them.
            self._manager = suitcase.utils.MultiFileManager(directory)
        else:
            # The user has given us their own Manager instance. Use that.
            self._manager = directory
        
        self._files = {}
        self._h5files = {}

        if dark is None:
            self._dark = None
        else:
            self._dark = extract_dark_images(dark)

        if mask_file is None:
            self._mask = None
        else:
            self._mask = np.load(mask_file)
        # to store the primary and baseline
        self._descriptors = {}
        
        self._baseline_triggered = False
        
    @property
    def artifacts(self):
        # The 'artifacts' are the manager's way to exposing to the user a
        # way to get at the resources that were created. For
        # `MultiFileManager`, the artifacts are filenames.  For
        # `MemoryBuffersManager`, the artifacts are the buffer objects
        # themselves. The Serializer, in turn, exposes that to the user here.
        #
        # This must be a property, not a plain attribute, because the
        # manager's `artifacts` attribute is also a property, and we must
        # access it anew each time to be sure to get the latest contents.
        return self._manager.artifacts
    

    def close(self):
        """
        Close all of the resources (e.g. files) allocated.
        """
        for f in self._h5files.values():
            f.close()
        self._manager.close()
        

    # These methods enable the Serializer to be used as a context manager:
    #
    # with Serializer(...) as serializer:
    #     ...
    #
    # which always calls close() on exit from the with block.

    def __enter__(self):
        return self

    def __exit__(self, *exception_details):
        self.close()

    # Each of the methods below corresponds to a document type. As
    # documents flow in through Serializer.__call__, the DocumentRouter base
    # class will forward them to the method with the name corresponding to
    # the document's type: RunStart documents go to the 'start' method,
    # etc.
    #
    # In each of these methods:
    #
    # - If needed, obtain a new file/buffer from the manager and stash it
    #   on instance state (self._files, etc.) if you will need it again
    #   later. Example:
    #
    #   filename = f'{self._templated_file_prefix}-primary.csv'
    #   file = self._manager.open('stream_data', filename, 'xt')
    #   self._files['primary'] = file
    #
    #   See the manager documentation below for more about the arguments to open().
    #
    # - Write data into the file, usually something like:
    #
    #   content = my_function(doc)
    #   file.write(content)
    #
    #   or
    #
    #   my_function(doc, file)

    def start(self, doc):
        # Fill in the file_prefix with the contents of the RunStart document.
        # As in, '{uid}' -> 'c1790369-e4b2-46c7-a294-7abfa239691a'
        # or 'my-data-from-{plan-name}' -> 'my-data-from-scan'
        self._templated_file_prefix = self._file_prefix.format(**doc)
        
        # The length of the scan is fixed at start time
        # We will use this later when we allocate space for the data
        self._n = doc['num_points']

        if 'fccd_intersection' in doc:
            self._fccd_intersection = doc['fccd_intersection']
            
        # Collect some relevant attributes
        # Some are non-essential and won't cause failure if they are
        # not found, whereas some are essential and will cause failure

        scan_id = doc['scan_id']
        group = doc['group'] if 'group' in doc else None

        sample_type = None
        sample_composition = None
        sample_lattice = None
        if 'sample' in doc:
            if 'type' in doc['sample']:
                sample_type = doc['sample']['type']
            if 'composition' in doc['sample']:
                sample_composition = doc['sample']['composition']
            if 'lattice' in doc['sample']:
                sample_lattice = doc['sample']['lattice']

        beamline_id = doc['beamline_id'] if 'beamline_id' in doc else None
        project = doc['project'] if 'project' in doc else None
        start_time = doc['time']

        
        
        # grab the motors being scanned with here:
        # I decided against this, because the CXI file requires that
        # the scan axes are oriented in a specific geometry. Allowing
        # files to be created with arbitrary motors as the scan motors,
        # we would need some way to automatically convert these into the
        # same uniform CXI coordinate space - It seemed better to restrict
        # to nanop_bx and nanop_bz, which we know
        #self._motors = doc['motors']

        # Next will be creating the various types of documents here.
        # 1) A "Raw" document with each and every individual exposure
        # 2) A "Flattened" document with one exposure per frame
        # The underlying file objects are created by the multifilemanager
        # so it deals with stuff like auto-creating the directory, and
        # the actual data writing is done with h5py so we make h5py
        # File objects too
        
        self._files['raw'] = self._manager.open('hdf5',
                        f'{self._templated_file_prefix}raw.cxi', 'xb')
        self._files['flat'] = self._manager.open('hdf5',
                        f'{self._templated_file_prefix}flat.cxi', 'xb')

        self._h5files = {key: h5py.File(f,'w') for key, f in self._files.items()}
        
        for f in self._h5files.values():
            f.create_dataset('cxi_version', data=160)  
            f.create_dataset('number_of_entries',data=1)
            e1 = f.create_group('entry_1')
            e1['run_id'] = scan_id

            # This is nonstandard but can't hurt to save it
            if project:
                e1['project'] = np.string_(project)
            if group:
                e1['group'] = np.string_(group)
                

            s1 = e1.create_group('sample_1')
            if sample_type:
                s1['name'] = np.string_(sample_type)
            if sample_composition:
                s1['description'] = np.string_(sample_composition)
            if sample_lattice:
                s1['unit_cell'] = np.string_(sample_lattice)

            # Just create the group and allocate space for translations
            g1 = s1.create_group('geometry_1')
            g1.create_dataset('translation', (self._n,3),dtype=np.float64)

            i1 = e1.create_group('instrument_1')
            i1['name'] = np.string_(beamline_id)

            d1 = i1.create_group('detector_1')
            x_pixel_size, y_pixel_size = self._pitch
            d1['x_pixel_size'] = x_pixel_size
            d1['y_pixel_size'] = y_pixel_size
            d1['distance'] = self._distance
            d1['translation'] = h5py.SoftLink('/entry_1/sample_1/geometry_1/translation')
            # We only add a mask attribute if a mask was given as a kwarg
            if self._mask is not None:
                d1['mask'] = self._mask
            
            so1 = i1.create_group('source_1')
            so1['name'] = np.string_('NSLS-II')

            data1 = e1.create_group('data_1')
            data1['translation'] = h5py.SoftLink('/entry_1/sample_1/geometry_1/translation')



    def descriptor(self, doc):
        # All we need to do here is store the descriptor information in
        # case we need to reference it later, and we also need to be able
        # to decode the uid references from the events
        self._descriptors[doc['uid']] = doc

    def event_page(self, doc):
        
        # The first kind of event is a "baseline" event encoding the parameters
        # that remain constant over the experiment.
        if self._descriptors[doc['descriptor']]['name'] == 'baseline' \
           or not self._baseline_triggered:
            
            self._baseline_triggered = True
            if doc['seq_num'][0] == 1:
            
                # Find and store the source energy and wavelength
                try:
                    energy = doc['data']['pgm_energy_readback'][0]
                except KeyError as e:
                    # For old scans, there was no energy readback
                    energy = doc['data']['pgm_energy_setpoint'][0]
                    
                energy = energy * self._harmonic * 1.60218e-19 # to J
                wavelength = 1.9864459e-25 / energy
                                
                # Get the TARDIS geometry
                try:
                    delta = doc['data']['tardis_delta'][0]
                    gamma = doc['data']['tardis_gamma'][0]
                except KeyError as e:
                    # On older files, they're called delta and gamma
                    delta = doc['data']['delta'][0]
                    gamma = doc['data']['gamma'][0]
                
                xps, yps = self._pitch
                basis_vectors = np.array([[0,-yps,0],[-xps,0,0]]).transpose()

                corner_position = np.dot(basis_vectors,-np.array(self._fccd_intersection))
                corner_position += np.array([0,0, self._distance])
                
                # The patterns of negative signs below are to emphasize
                # the correct rotation matrices and the direction
                # (clockwise or counterclockwise) that rotation is defined
                # in for the various diffractometer directions
                rotation_delta = np.array([[1,0,0],
                                           [0,np.cos(np.deg2rad(-delta)),-np.sin(np.deg2rad(-delta))],
                                           [0,np.sin(np.deg2rad(-delta)),np.cos(np.deg2rad(-delta))]]) 
                rotation_gamma = np.array([[np.cos(np.deg2rad(gamma)),0,np.sin(np.deg2rad(gamma))],
                                           [0,1,0],
                                           [-np.sin(np.deg2rad(gamma)),0,np.cos(np.deg2rad(gamma))]]) 
                
                rotation = np.dot(rotation_delta,rotation_gamma)
                basis_vectors = np.dot(rotation,basis_vectors)
                corner_position = np.dot(rotation,corner_position)
                
    
                s_time = doc['time'][0]
                # cxi format requires nanosecond precision
                time_str = datetime.utcfromtimestamp(s_time).isoformat()+'000'

                for f in self._h5files.values():
                    e1 = f['entry_1']
                    d1 = e1['instrument_1/detector_1']
                    d1['corner_position'] = corner_position
                    d1['delta'] = delta
                    d1['gamma'] = gamma
                    d1['basis_vectors'] = basis_vectors
                    e1['start_time'] = np.string_(time_str)
                    
                    so1 = e1['instrument_1/source_1']
                    so1['energy'] = energy
                    so1['wavelength'] = wavelength

            elif doc['seq_num'][0] == 2:
                e_time = doc['time'][0]
                # cxi format requires nanosecond precision
                time_str = datetime.utcfromtimestamp(e_time).isoformat()+'000'
                for f in self._h5files.values():
                    f['entry_1/end_time'] = np.string_(time_str)

                
        # Don't move to elif, because for old scans with no baseline document,
        # we need to use the first primary document for both.
        if self._descriptors[doc['descriptor']]['name'] == 'primary':
            # First, we have to define the data and translation arrays
            # when the first event comes in so we know their types and shapes
            if doc['seq_num'][0] == 1:
                example_im = process_images_csx(doc['data']['fccd_image'])
                raw_shape = (self._n,) + example_im.shape
                flat_shape = (self._n,) + example_im.shape[-2:]
                raw_d1 = self._h5files['raw']['/entry_1/instrument_1/detector_1']
                flat_d1 = self._h5files['flat']['/entry_1/instrument_1/detector_1']
                raw_d1.create_dataset('data',raw_shape,dtype=example_im.dtype)
                flat_d1.create_dataset('data',flat_shape,dtype=example_im.dtype)
                self._h5files['raw']['entry_1/data_1/data'] = h5py.SoftLink('/entry_1/instrument_1/detector_1/data')
                self._h5files['flat']['entry_1/data_1/data'] = h5py.SoftLink('/entry_1/instrument_1/detector_1/data')
                # potentially add axis labels here and to the translations?

            # Now we actually have to deal with the incoming image data

            raw_im = process_images_csx(doc['data']['fccd_image'],
                                        dark=self._dark)
            flat_im = np.nanmean(raw_im,axis=0)
            raw_d1 = self._h5files['raw']['/entry_1/instrument_1/detector_1']
            flat_d1 = self._h5files['flat']['/entry_1/instrument_1/detector_1']
            idx = doc['seq_num'][0]-1
            print(idx)
            raw_d1['data'][idx,:,:,:] = raw_im
            flat_d1['data'][idx,:,:] = flat_im

            # And the incoming translation data
            # The negative signs and axis permutation is done to ensure that
            # the motion axes coincide with the assumed axes of the CXI
            # files
            pos_x = -doc['data']['nanop_bx'][0] * 1e-3 # convert to m
            pos_y = -doc['data']['nanop_bz'][0] * 1e-3 # convert to m
            if 'nanop_by' in doc['data']:
                pos_z = doc['data']['nanop_by'][0] * 1e-3 # convert to m
            else:
                pos_z = np.zeros_like(pos_x)
                
            raw_d1['translation'][idx,:] = np.array([pos_x,pos_y,pos_z])
            flat_d1['translation'][idx,:] = np.array([pos_x,pos_y,pos_z])
            


    def stop(self, doc):
        pass



def process_images_csx(images, dark=None, flat=None, gain=(1,4,8)):
    if type(images) == type([]):
        # It's odd, but in v1 the images seem to come out not in a list,
        # but in v2 they seem to come out in a length-1 list. Not sure why.
        images = images[0]

    ims = correct_images(images, dark=dark, flat=flat, gain=gain)
    if ims.shape[-2:] == (960,960):
        # Some old images come out with no final dimension and
        # already trimmed to 960x960, this uses that size
        #print('old_style')
        return images_trim(rotate90(ims, 'cw'),lcol=478,col_shift=4, trim_outer=False)[...,::-1]

    # We flip the last axis to get the image into the most standard
    # form for CXI files, which also meshes with the explicit coordinates
    # that we encode into the files in this suitcase
    return images_trim(rotate90(ims, 'cw'))[...,::-1]

        
def images_trim(images, lcol = 486, col_shift = 28, trim_outer=True):
    """Gets rid of the fake central set of pixels in the raw images.
    come to think of it - why do these exist in the first place??"""
    rcol = lcol + col_shift
    im_dim = np.shape(images)
    images = np.delete(images,range(lcol,rcol),axis=-1)

    if trim_outer:
        im_shift = (im_dim[-1]-im_dim[-2]-col_shift)/2
        return images[...,int(im_shift):-int(im_shift)]
    else:
        return images

def extract_dark_images(dark):
    # We assume that all images are for the background
    # TODO : Perhaps we can loop over the generator
    # If we want to do something lazy

    d_ims = []
    for d in dark:
        if hasattr(d,'primary'):
            # This is for the v2 interface
            raw = d.primary.read()['fccd_image'][0]
        else:
            # And this is for the v1 interface
            raw = next(d.data('fccd_image'))

        corrected = correct_images(raw, gain=(1, 1, 1))
        d_ims.append(np.nanmean(corrected,axis=0))

    return d_ims

extern crate image;

use std::fs::File;
use std::io::{Read, Seek, SeekFrom, self};
use std::iter;
use std::ops::Range;
use std::path::Path;

use cast::From;
use self::image::{ImageBuffer, Luma};
use super::byteorder::{BigEndian, ReadBytesExt};

use ::prelude::*;

use nn::F;

type cMat = ::Mat<F, ::order::Col>;
type rMat = ::Mat<F, ::order::Row>;

/// A set of images in compressed format
pub struct Images {
    data: Vec<u8>,
    height: u32,
    img_size: usize,
    size: u32,
    width: u32,
}

impl Images {
    /// Loads a subset of the images stored in `path`
    pub fn load<P>(path: P, subset: Range<u32>) -> io::Result<Images> where P: AsRef<Path> {
        Images::load_(path.as_ref(), subset)
    }

    fn load_(path: &Path, Range { start, end }: Range<u32>) -> io::Result<Images> {
        /// Magic number expected in the header
        const MAGIC: u32 = 2051;

        assert!(start < end);

        let mut file = try!(File::open(path));

        // Parse the header: MAGIC NIMAGES NROWS NCOLS
        assert_eq!(try!(file.read_u32::<BigEndian>()), MAGIC);
        let nimages = try!(file.read_u32::<BigEndian>());
        let nrows = try!(file.read_u32::<BigEndian>());
        let ncols = try!(file.read_u32::<BigEndian>());

        assert!(end <= nimages);

        let img_size = usize::from(nrows).checked_mul(usize::from(ncols)).unwrap();
        let buf_size = img_size * usize::from(end - start);
        let mut buf: Vec<_> = iter::repeat(0).take(buf_size).collect();

        try!(file.seek(SeekFrom::Current(i64::from(img_size * usize::from(start)).unwrap())));

        assert_eq!(try!(file.read(&mut buf)), buf_size);

        Ok(Images {
            data: buf,
            height: nrows,
            img_size: img_size,
            size: end - start,
            width: ncols,
        })
    }

    /// Returns the `i`th image stored in the set, in unrolled form
    fn get_img(&self, i: u32) -> &[u8] {
        let i = usize::from(i);
        let sz = self.img_size;
        let start = i * sz;
        let end = start + sz;

        &self.data[start..end]
    }

    /// Returns the number of pixels per image
    pub fn num_pixels(&self) -> u32 {
        self.width * self.height
    }

    /// Returns the size of this set
    pub fn size(&self) -> u32 {
        self.size
    }

    /// Return the images as a data set that can be fed to a neural network
    ///
    /// The returned matrix has dimensions m-by-(n+1), where
    ///
    /// m: Number of images
    /// n: Number of pixels per image
    ///
    /// Each row of the matrix is an "unrolled" image where each element of the row represents the
    /// brightness (in the 0.0 - 1.0 range) of a single pixel
    pub fn to_dataset(&self) -> Box<rMat> {
        let mut m = rMat::ones((self.size, self.width * self.height + 1));

        for (mut row, img) in (&mut *m).rows_mut().zip(self.data.chunks(self.img_size)) {
            for (e, &brightness) in row.iter_mut().skip(1).zip(img) {
                *e = F::from(brightness) / F::from(u8::max_value())
            }
        }

        m
    }

    /// Arranges `these` images (up to 100) in a matrix, and stores the resulting image in `path`
    pub fn save<I, P>(&self, these: I, path: P) -> io::Result<()> where
        I: IntoIterator<Item=u32>,
        P: AsRef<Path>,
    {
        self.save_(these.into_iter(), path.as_ref())
    }

    fn save_<I>(&self, mut these: I, path: &Path) -> io::Result<()> where
        I: Iterator<Item=u32>
    {
        type Mat = ::Mat<u8, ::order::Row>;

        const NROWS: u32 = 10;
        const NCOLS: u32 = 10;

        let (height, width) = (NROWS * self.height, NCOLS * self.width);
        let mut buf = Mat::from_elem((height, width), 0);

        'out: for buf in buf.hstripes_mut(self.height) {
            for buf in buf.vstripes_mut(self.width) {
                let i = if let Some(i) = these.next() { i } else { break 'out };

                buf[..] = Mat::reshape(self.get_img(i), (self.height, self.width));
            }
        }

        let img = ImageBuffer::<Luma<u8>, _>::from_raw(width, height, buf.as_ref()).unwrap();

        img.save(path)
    }
}

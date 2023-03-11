const IMAGE_SIZE: usize = 2000; // In pixels (x by x png)
const NUM_THREADS: usize = 4;

const SAMPLES: usize = 2000; // Per pixel
const ITERATIONS: [usize; 3] = [10000, 1000, 100]; // Iterations for [red, green, blue] channels
const WEIGHTS: [f64; 3] = [1., 1.2, 1.5]; // Gamma coeficeint for [red, green, blue]

fn main() -> Result<(), impl std::error::Error> {
    let image_data = Arc::new([
        // Mutexes can't be cloned, [val;3]
        Mutex::new([[0.0; IMAGE_SIZE]; IMAGE_SIZE]),
        Mutex::new([[0.0; IMAGE_SIZE]; IMAGE_SIZE]),
        Mutex::new([[0.0; IMAGE_SIZE]; IMAGE_SIZE]),
    ]);
    let mut handels = Vec::with_capacity(NUM_THREADS);
    let sampled = Arc::new(AtomicUsize::new(0));
    for _ in 0..NUM_THREADS {
        let sampled = sampled.clone();
        let data = image_data.clone();

        handels.push(thread::spawn(move || {
            render(data, sampled);
        }));
    }

    for handel in handels {
        handel.join().unwrap();
    }

    let mut image = ImageBuffer::<Rgb<u16>, Vec<u16>>::new(IMAGE_SIZE as u32, IMAGE_SIZE as u32);
    let red_image = image_data[0].lock().unwrap();
    let green_image = image_data[1].lock().unwrap();
    let blue_image = image_data[2].lock().unwrap();

    for x in 0..IMAGE_SIZE {
        for y in 0..IMAGE_SIZE {
            let pixel = Rgb::from([
                red_image[x][y] as u16,
                green_image[x][y] as u16,
                blue_image[x][y] as u16,
            ]);
            image.put_pixel(x as u32, y as u32, pixel);
        }
    }
    image.save("Result.png")
}

fn render(data: Arc<[Mutex<ChannelData>; 3]>, sampled: Arc<AtomicUsize>) {
    let mut rng = thread_rng();
    while sampled.load(ORDERING) < SAMPLES {
        println!(
            "Computing {} of {SAMPLES} samples",
            sampled.fetch_add(1, ORDERING) + 1
        );
        // image quality should be resolution indepentent
        for _ in 0..IMAGE_SIZE * IMAGE_SIZE {
            for (channel, channel_iterations, weight) in
                izip!(data.iter(), ITERATIONS.iter(), WEIGHTS.iter())
            {
                accumlate_samples(channel, channel_iterations, weight, &mut rng);
            }
        }
    }
}

type ChannelData = [[f64; IMAGE_SIZE]; IMAGE_SIZE];

fn accumlate_samples(
    channel: &Mutex<ChannelData>,
    channel_iterations: &usize,
    weight: &f64,
    rng: &mut ThreadRng,
) {
    let c = Complex::new(rng.gen(), rng.gen());
    let c = (c - 0.5) * 2. * SIMULATION_BOUNDS;
    let mut entered = Vec::new();
    let mut z = Complex::new(0., 0.);
    for _ in 0..*channel_iterations {
        z = z * z + c;
        if z.i.abs() > 2. && z.r.abs() > 2. {
            break;
        }
        entered.push(z);
    }
    let mut data = channel.lock().unwrap();
    for c in entered {
        let coord = complex_to_image(c);
        if let Some(row) = data.get_mut(coord.0) {
            if let Some(brightnes) = row.get_mut(coord.1) {
                *brightnes += weight
            }
        }
    }
}

use image::ImageBuffer;
use image::Rgb;
use itertools::izip;
use rand::prelude::*;
use std::ops::Add;
use std::{
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    },
    thread,
};
const ORDERING: Ordering = Ordering::SeqCst;
const SIMULATION_BOUNDS: f64 = 2.;

fn complex_to_image(complex: Complex) -> (usize, usize) {
    const SIZE_AS_F64: f64 = IMAGE_SIZE as f64;
    let i = complex.r * 0.5 / SIMULATION_BOUNDS + 0.5;
    let j = complex.i * 0.5 / SIMULATION_BOUNDS + 0.5;
    return ((i * SIZE_AS_F64) as usize, (j * SIZE_AS_F64) as usize);
}

#[derive(Clone, Default, Copy, Debug)]
pub struct Complex {
    pub r: f64,
    pub i: f64,
}

impl Complex {
    pub const fn new(r: f64, i: f64) -> Self {
        Self { r, i }
    }

    pub fn len(&self) -> f64 {
        (self.r * self.r + self.i * self.i).sqrt()
    }
}

impl Add for Complex {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            r: self.r + rhs.r,
            i: self.i + rhs.i,
        }
    }
}

impl Sub for Complex {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            r: self.r - rhs.r,
            i: self.i - rhs.i,
        }
    }
}

use std::ops::Mul;
impl Mul for Complex {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self {
            r: self.r * rhs.r - self.i * rhs.i,
            i: self.r * rhs.i + self.i * rhs.r,
        }
    }
}

impl Add<f64> for Complex {
    type Output = Self;
    fn add(self, rhs: f64) -> Self {
        Self {
            r: self.r + rhs,
            i: self.i + rhs,
        }
    }
}

use std::ops::Sub;
impl Sub<f64> for Complex {
    type Output = Self;
    fn sub(self, rhs: f64) -> Self {
        Self {
            r: self.r - rhs,
            i: self.i - rhs,
        }
    }
}

impl Mul<f64> for Complex {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        Self {
            r: self.r * rhs,
            i: self.i * rhs,
        }
    }
}

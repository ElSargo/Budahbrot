#![feature(const_trait_impl)]
use glam::{DVec2, UVec2};
use image::{ImageBuffer, Rgb};
use rand::{thread_rng, Rng};
use std::sync::atomic::{AtomicUsize, Ordering};
const NUM_WORKERS: usize = 20;
const RESOLUTION: usize = 450;
const SAMPLES: usize = 50;
const ITERATIONS: [usize; 3] = [5000, 500, 50];
const WEIGHTS: [f64; 3] = [15.0, 15.0, 10.0];

fn main() {
    let data = Box::new([ATOMIC_NESTED_ARRAY_INIT; 3]);
    let sample_count = AtomicUsize::new(0);
    std::thread::scope(|s| {
        for _ in 0..NUM_WORKERS {
            s.spawn(|| {
                let mut rng = thread_rng();
                // Make it resolution independent
                while sample_count.fetch_add(1, Ordering::Relaxed) < TOTAL_SAMPLES {
                    let mut stack = [Cmplx::ZERO; MAX_ITER + 1];
                    let mut z = Cmplx::ZERO;
                    let c = Cmplx::new(rng.gen_range(-2.0..2.0), rng.gen_range(-2.0..2.0));
                    let mut i = 0;
                    while i <= MAX_ITER && z.length() < 2.0 {
                        z = z * z + c;
                        stack[i] = z;
                        i += 1;
                    }
                    for channel in 0..3 {
                        let iterations = ITERATIONS[channel];
                        if i > ITERATIONS[channel] || i == 0 {
                            continue;
                        }
                        for idx in 0..(iterations.min(i)) {
                            let coord = cmplx_to_pixel(stack[idx]);
                            match data[channel]
                                .get(coord.x as usize)
                                .and_then(|row| row.get(coord.y as usize))
                            {
                                Some(pixel) => {
                                    pixel.fetch_add(IWEIGHTS[channel], Ordering::Relaxed)
                                }
                                None => break,
                            };
                        }
                    }
                }
            });
        }
    });

    let mut image: ImageBuffer<Rgb<u16>, Vec<_>> =
        ImageBuffer::new(RESOLUTION as u32, RESOLUTION as u32);
    let red_channel = &data[0];
    let green_channel = &data[1];
    let blue_channel = &data[2];

    let pixel = |x: usize, y: usize| -> [u16; 3] {
        [
            (red_channel[x][y].load(Ordering::Relaxed) * u16::MAX as f64) as u16,
            (green_channel[x][y].load(Ordering::Relaxed) * u16::MAX as f64) as u16,
            (blue_channel[x][y].load(Ordering::Relaxed) * u16::MAX as f64) as u16,
        ]
    };
    for (x, y) in (0..RESOLUTION)
        .map(|x| (0..RESOLUTION).map(move |y| (x, y)))
        .flatten()
    {
        *image.get_pixel_mut(x as u32, y as u32) = Rgb::from(pixel(x, y));
    }
    if let Err(e) = image.save("result.png") {
        println!("{:?}", e);
    }
}

#[derive(Default, Debug, PartialEq, Copy, Clone)]
struct Cmplx {
    data: DVec2,
}

impl Cmplx {
    const ZERO: Self = Self::new(0.0, 0.0);
    const fn new(x: f64, y: f64) -> Self {
        Self {
            data: DVec2 { x, y },
        }
    }

    fn length(&self) -> f64 {
        self.data.length()
    }
}

impl std::ops::Mul for Cmplx {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self::new(
            self.data.x * rhs.data.x - self.data.y * rhs.data.y,
            self.data.x * rhs.data.y + self.data.y * rhs.data.x,
        )
    }
}

impl std::ops::Mul<f64> for Cmplx {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        Self {
            data: self.data * rhs,
        }
    }
}

impl std::ops::Add for Cmplx {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        (self.data + rhs.data).into()
    }
}

impl From<DVec2> for Cmplx {
    fn from(value: DVec2) -> Self {
        Self { data: value }
    }
}
impl From<[f64; 2]> for Cmplx {
    fn from(value: [f64; 2]) -> Self {
        Self {
            data: DVec2 {
                x: value[0],
                y: value[1],
            },
        }
    }
}

impl Into<DVec2> for Cmplx {
    fn into(self) -> DVec2 {
        self.data
    }
}

fn cmplx_to_pixel(cmplx: Cmplx) -> UVec2 {
    const FRESOLUTION: f64 = RESOLUTION as f64;
    let neg_two_to_two: DVec2 = cmplx.into();
    let normal = neg_two_to_two * 0.25 + 0.5;
    let pix = normal * FRESOLUTION;
    UVec2 {
        x: pix.x as u32,
        y: pix.y as u32,
    }
}

const ATOMIC_ZERO: atomic_float::AtomicF64 = atomic_float::AtomicF64::new(0.0);
const ATOMIC_ARRAY_INIT: [atomic_float::AtomicF64; RESOLUTION] = [ATOMIC_ZERO; RESOLUTION];
const ATOMIC_NESTED_ARRAY_INIT: [[atomic_float::AtomicF64; RESOLUTION]; RESOLUTION] =
    [ATOMIC_ARRAY_INIT; RESOLUTION];
const MAX_ITER: usize = ITERATIONS[0].max(ITERATIONS[1].max(ITERATIONS[2]));
const TOTAL_SAMPLES: usize = NUM_WORKERS * SAMPLES * RESOLUTION * RESOLUTION;
const IWEIGHTS: [f64; 3] = [
    1.0 / (WEIGHTS[0] * SAMPLES as f64 * NUM_WORKERS as f64),
    1.0 / (WEIGHTS[1] * SAMPLES as f64 * NUM_WORKERS as f64),
    1.0 / (WEIGHTS[2] * SAMPLES as f64 * NUM_WORKERS as f64),
];

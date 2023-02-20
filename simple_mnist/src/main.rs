use burn_autodiff::ADBackendDecorator;
use burn_ndarray::{NdArrayBackend, NdArrayDevice};
use simple_mnist::training;

fn main() {
    let device = NdArrayDevice::Cpu;
    training::run::<ADBackendDecorator<NdArrayBackend<f32>>>(device);
}

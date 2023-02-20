use crate::data::MNISTBatch;
use burn::{
    module::{Module, Param},
    nn::conv::{Conv2d, Conv2dConfig, Conv2dPaddingConfig},
    nn::loss::CrossEntropyLoss,
    nn::pool::{MaxPool2d, MaxPool2dConfig},
    nn::{Dropout, DropoutConfig, Linear, LinearConfig},
    tensor::activation::relu,
    tensor::{
        backend::{ADBackend, Backend},
        Tensor,
    },
    train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep},
};

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    conv1: Param<Conv2d<B>>,
    conv2: Param<Conv2d<B>>,
    dropout1: Dropout,
    dropout2: Dropout,
    linear1: Param<Linear<B>>,
    linear2: Param<Linear<B>>,
    max_pool: MaxPool2d,
}

impl<B: Backend> Model<B> {
    pub fn new() -> Self {
        Self {
            conv1: Param::new(Conv2d::new(
                &Conv2dConfig::new([1, 32], [3, 3]),
            )),
            conv2: Param::new(Conv2d::new(
                &Conv2dConfig::new([32, 64], [3, 3]),
            )),
            dropout1: Dropout::new(&DropoutConfig::new(0.25)),
            dropout2: Dropout::new(&DropoutConfig::new(0.5)),
            linear1: Param::new(Linear::new(&LinearConfig::new(9216, 128))),
            linear2: Param::new(Linear::new(&LinearConfig::new(128, 10))),
            max_pool: MaxPool2d::new(&MaxPool2dConfig::new(64, [2, 2]).with_strides([2, 2])),
        }
    }
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch_size, heigth, width] = input.dims();
        let x = input.reshape([batch_size, 1, heigth, width]).detach();
        let x = self.conv1.forward(x);
        let x = relu(&x);
        let x = self.conv2.forward(x);
        let x = relu(&x);
        let x = self.max_pool.forward(x);
        let x = self.dropout1.forward(x);
        let x = x.reshape([batch_size, 9216]);
        let x = self.linear1.forward(x);
        let x = relu(&x);
        let x = self.dropout2.forward(x);
        let out = self.linear2.forward(x);
        out
    }
    pub fn forward_classification(&self, item: MNISTBatch<B>) -> ClassificationOutput<B> {
        let targets = item.targets;
        let output = self.forward(item.images);
        let loss = CrossEntropyLoss::new(10, None);
        let loss = loss.forward(&output, &targets);
        ClassificationOutput {
            loss,
            output,
            targets,
        }
    }
}

impl<B: ADBackend> TrainStep<B, MNISTBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, item: MNISTBatch<B>) -> TrainOutput<B, ClassificationOutput<B>> {
        let item = self.forward_classification(item);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<MNISTBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, item: MNISTBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(item)
    }
}

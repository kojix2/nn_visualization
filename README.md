# nn_visualization
nn : A multilayer neural network written in Ruby NArray.
visualization : druby + Ruby/Tk (Plotchart)

Although Ruby codes are considered slow, this example can train your neural network with practical speed and predict MNIST handwritten digits in parallel processes.

## Install
1. Install Numo::NArray
2. Install Numo::Linalg
3. Install parallels

## Usage
```bash
$ ruby example_mnist.rb
```

```ruby
# Make a neural netwrok
n = NN.new{
  layer 784, 100, :tanh
  layer 100, 10,  :sigmoid
  # You can add the layers as much as you like. 
}

# Train your neural network and predict digits.
10.times do |i|
  n.train(train_images, train_labels, 0.005, 0.0005)
  evaluate n, test_images, test_labels
end
```

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
$ ruby drb.rb &
$ ruby client_gui.rb &
$ ruby example_mnist.rb
...
$ pkill ruby # kill processes
```

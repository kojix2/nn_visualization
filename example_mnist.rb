require_relative "nn" 
require_relative "download_mnist.rb"
require "drb/drb"
require "zlib"
require "optparse"

test = nil
opt = OptionParser.new
opt.on('-t'){|v| test = v}
opt.parse!(ARGV)

# http://d.hatena.ne.jp/n_shuyo/20090913/mnist
def read_images(file_path)
  Zlib::GzipReader.open(file_path) do |f|
    magic, n_images = f.read(8).unpack('N2')
    raise "This is not MNIST image file" if magic != 2051
    n_rows, n_cols = f.read(8).unpack('N2')
    Array.new(n_images) do 
      f.read(n_rows * n_cols).unpack('C*')
    end
  end
end

# http://d.hatena.ne.jp/n_shuyo/20090913/mnist 
def read_labels(file_path)
  Zlib::GzipReader.open(file_path) do |f|
    magic, n_labels = f.read(8).unpack('N2')
    raise "This is not MNIST label file" if magic != 2049
    f.read(n_labels).unpack('C*')
  end
end

unless test
  puts "load train-images..."
  train_images = read_images 'train-images-idx3-ubyte.gz'
  puts "load train-labels..."
  train_labels = read_labels 'train-labels-idx1-ubyte.gz'
end

puts "load test-images..."
test_images = read_images 't10k-images-idx3-ubyte.gz'
puts "load test-labels..."
test_labels = read_labels 't10k-labels-idx1-ubyte.gz'

def get_label(n)
  na = Numo::Int8.zeros(10)
  na[n] = 10 # trick
  na
end

unless test
  puts "convert train images and labels..."
  train_images.map!{|ti| Numo::DFloat[*ti] / 256.0 }
  train_labels.map!{|tl| get_label(tl) }
end

puts "convert test images and labels..."
test_images.map!{|ti| Numo::DFloat[*ti] / 256.0 }
test_labels.map!{|tl| get_label(tl) }

puts "connect to druby://localhost:12345"
@dnn = DRbObject.new_with_uri('druby://localhost:12345')

def evaluate(n, test_images, test_labels)
  correct = 0
  incorrect = 0

  results = n.predict_multiprocess(test_images)
  results.zip(test_labels).each do |result, t_lab|
    result = result.first
    result = result.index(result.max)
    label = t_lab.to_a.index(10)
    if result == label
      correct += 1
    else
      incorrect += 1
    end
  end
  puts
  puts "correct #{correct}"
  puts "incorrect #{incorrect}"

  @dnn.accuracy_rate = 100 * correct / (correct + incorrect).to_f
end

# MAIN


# Make a neural netwrok
n = NN.new{
  layer 784, 100, :tanh
  layer 100, 10,  :sigmoid
  # You can add the layers as much as you like. 
}

# Train your neural network and predict digits.
10.times do |i|
  if test
    n.train(test_images, test_labels, 0.005, 0.0005)
  else
    n.train(train_images, train_labels, 0.005, 0.0005)
  end
  evaluate n, test_images, test_labels

  ws = []
  n.each_with_index do |layer, index|
    ws << layer.w
  end

  @dnn.w1 = ws[0].to_string
  @dnn.w1_shape = ws[0].shape
  @dnn.w2 = ws[1].to_string
  @dnn.w2_shape = ws[1].shape
end

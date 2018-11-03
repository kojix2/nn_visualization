require 'numo/narray'
require 'numo/linalg'
require 'parallel'

module ActivationFunctions
  def linear(x)
    x
  end

  def dlinear(x)
    x
  end

  def sigmoid(x)
    1.0 / (1.0 + Numo::NMath.exp(-x))
  end

  def dsigmoid(x)
    x * (-x + 1.0)
  end

  def tanh(x)
    Numo::NMath.tanh(x)
  end

  def dtanh(x)
    -(x**2) + 1.0
  end

  #  def relu(x)       x[x < 0] = 0.0; x                   end
  #  def drelu(x)      x[x > 0] = 1.0; x[x < 0] = 0.0; x   end
  #  def softmax(x)
  #    e = Numo::NMath.exp(x - x.max)
  #    e /= e.sum
  #  end
  def matmul(a, w)
    Numo::Linalg.matmul(a, w)
  end
end

# Neural Network Class
class NN
  include Enumerable

  class Layer
    include Numo
    include ActivationFunctions

    def initialize(_in, _out, _func)
      @a = DFloat.new(1, _in + 1).fill(1.0)
      @w = DFloat.new(_in + 1, _out).rand(-0.2, 0.2)
      @c = DFloat.new(_in + 1, _out).fill(1.0)
      @func = _func
      @dfunc = ('d' << _func.to_s).to_sym
      @size = _in
      @n = 0; @m = 0
    end

    def forward(input)
      @a[0...-1] = input
      @o = send(@func, matmul(@a, @w))
    end

    def backward(error_array)
      @deltas = send(@dfunc, @o) * error_array
      matmul(@deltas, @w.transpose)[true, 0...-1] # Parhaps "dot" and "sum" may be faster
    end

    # Momentum
    def update
      change = get_change
      @w = @w + change * @n + @c * @m
      @c = change
    end

    def get_change
      @a.transpose * @deltas
    end

    attr_accessor :m, :n, :a, :w, :func
    attr_reader :size
  end

  def initialize(&block)
    @layers = []
    @ao = nil
    instance_eval(&block)
  end

  def each
    @layers.each do |item|
      yield item
    end
  end

  def forward(input) # inputs should be instance of NArray
    raise 'wrong number of inputs' if input.size != @layers[0].size

    @ao = @layers.inject(input) { |memo, l| l.forward memo }
  end

  def backpropagate(targets)
    raise 'wrong number of target values' if targets.size != @ao.size

    @layers.reverse.inject(targets - @ao) { |memo, l| l.backward memo }
    @layers.map(&:update)
    ((-@ao + targets)**2).sum / 2.0 # Error
  end

  # Parallel (CPU x 3)
  def predict_multiprocess(inputs)
    Parallel.map(inputs, in_processes: 3) do |input|
      @layers.inject(input) { |memo, l| l.forward memo }.to_a
    end
  end

  def train(inputs, outputs, n, m)
    # n: learning rate, m: momentum factor
    @layers.each { |l| l.m = m; l.n = n }

    inputs_size = inputs.size
    raise 'inputs.size != outputs.size' if inputs_size != outputs.size

    inputs_size.times.inject(0.0) do |error, t|
      pgbar(t, inputs_size)
      forward(inputs[t])
      error += backpropagate(outputs[t])
    end # return error
  end

  private

  # create a layer
  def layer(_in, _out, _func)
    @layers << Layer.new(_in, _out, _func)
  end

  def pgbar(count, max)
    print "working... #{count + 1}/#{max}\r" if count % 1000 == 0
    STDOUT.flush
  end
end

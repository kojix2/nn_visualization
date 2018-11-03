require 'drb/drb'

# Front object

class NNStatus
  def initialize
    @accuracy_rate
    @w1
    @w1_shape
    @w2
    @w2_shape
  end
  attr_accessor :accuracy_rate
  attr_accessor :w1, :w1_shape
  attr_accessor :w2, :w2_shape
end

DRb.start_service('druby://localhost:12345', NNStatus.new)

DRb.thread.join

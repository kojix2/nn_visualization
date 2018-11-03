require 'tk'
require 'tkextlib/tcllib/plotchart'
require 'drb/drb'
require 'open3'
require 'numo/narray'

schart = nil

test_text = nil
ppm_image1 = ("P5\n280 280\n255\n" + Numo::UInt8.new(784, 100).seq.to_string).encode('ascii-8bit')
ppm_image2 = ("P5\n10 10\n255\n" + Numo::UInt8.new(10, 10).seq.to_string).encode('ascii-8bit')

tkimg1 = TkPhotoImage.new(data: ppm_image1)
tkimg2 = TkPhotoImage.new(data: ppm_image2)

dnn = DRbObject.new_with_uri('druby://localhost:12345')

# GUI
TkRoot.new(title: 'nn_visualization') do |root|
  TkLabel.new(root) do
    text 'Neural network visualization with Ruby'
    font TkFont.new(family: :times, size: 15)
    anchor :w
    pack fill: :x, padx: 10
  end

  TkCanvas.new(root) do |canvas|
    width 400
    height 160
    schart = Tk::Tcllib::Plotchart::Stripchart.new(canvas, [0.0, 10.0, 1.0], [80.0, 100.0, 2.0]) do
      title 'Accuracy rate'
    end
    pack
  end

  TkLabel.new(root) do
    text 'Weight 1'
    font TkFont.new(family: :times, size: 15)
    anchor :w
    pack fill: :x, padx: 10
  end

  TkLabel.new(root) do
    image tkimg1
    pack
  end

  TkLabel.new(root) do
    text 'Weight 2'
    font TkFont.new(family: :times, size: 15)
    anchor :w
    pack fill: :x, padx: 10
  end

  TkLabel.new(root) do
    image tkimg2
    pack
  end
end

counter = 0

TkTimer.start(500) do
  accuracy_rate = dnn.accuracy_rate
  dnn.accuracy_rate = nil
  if accuracy_rate
    w1 = dnn.w1
    w1_shape = dnn.w1_shape
    w2 = dnn.w2
    schart.plot('accuracy_rate', counter, accuracy_rate)
    schart.dataconfig('accuracy_rate', colour: :cyan, filled: :down, fillcolour: :blue)

    p w1[0..10]
    p w1_shape

    nw1 = Numo::DFloat.from_string(w1).reshape(*w1_shape)

    puts "w1.mean #{nw1.mean}"
    puts "w1.max #{nw1.max}"
    puts "w1.min #{nw1.min}"

    wdisplay = ((nw1 - nw1.min) / (nw1.max - nw1.min)) * 255
    wdisplay = Numo::UInt8.cast(wdisplay)

    wdisplay = wdisplay[0..783, true]
    wdisplay = wdisplay.reshape(28, 28, 10, 10)
    wdisplay = wdisplay.transpose(0, 2, 1, 3)
    wdisplay = wdisplay.reshape(280, 280)

    ppm_image1 = ("P5\n280 280\n255\n" + wdisplay.to_string).encode('ascii-8bit')

    tkimg1.data = ppm_image1
    Tk.update

    counter += 1

  end
end

Tk.mainloop

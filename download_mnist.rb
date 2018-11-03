require 'open-uri'

mnist_files = [
  'train-images-idx3-ubyte.gz',
  'train-labels-idx1-ubyte.gz',
  't10k-images-idx3-ubyte.gz',
  't10k-labels-idx1-ubyte.gz'
]

mnist_files.each do |filename|
  next if File.exist?(filename)

  print "downloadng #{filename}. Please wait.."
  data = open('http://yann.lecun.com/exdb/mnist/' + filename).read
  File.binwrite(filename, data)
  puts '..done'
end

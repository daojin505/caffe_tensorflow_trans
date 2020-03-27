from kaffe.tensorflow import Network

class TRANCOS_CCNN(Network):
    def setup(self):
        (self.feed('data_s0')
             .conv(7, 7, 32, 1, 1, relu=False, name='conv1')
             .batch_normalization(relu=True, name='conv1_bn')
             .max_pool(2, 2, 2, 2, name='pool1')
             .conv(7, 7, 32, 1, 1, relu=False, name='conv2')
             .batch_normalization(relu=True, name='conv2_bn')
             .max_pool(1, 1, 1, 1, name='pool2')
             .conv(2, 2, 32, 2, 2, relu=False, name='resx1_match_conv')
             .batch_normalization(name='resx1_match_conv_bn'))

        (self.feed('pool1')
             .conv(1, 1, 32, 1, 1, relu=False, name='resx1_conv1')
             .batch_normalization(relu=True, name='resx1_conv1_bn')
             .conv(3, 3, 32, 1, 1, group=32, relu=False, name='resx1_conv2')
             .batch_normalization(relu=True, name='resx1_conv2_bn')
             .conv(1, 1, 32, 2, 2, relu=False, name='resx1_conv3')
             .batch_normalization(name='resx1_conv3_bn'))

        (self.feed('resx1_conv3_bn', 
                   'resx1_match_conv_bn')
             .add(name='resx1_elewise')
             .relu(name='resx1_elewise_relu')
             .conv(1, 1, 32, 1, 1, relu=False, name='resx2_conv1')
             .batch_normalization(relu=True, name='resx2_conv1_bn')
             .conv(3, 3, 32, 1, 1, group=32, relu=False, name='resx2_conv2')
             .batch_normalization(relu=True, name='resx2_conv2_bn')
             .conv(1, 1, 32, 1, 1, relu=False, name='resx2_conv3')
             .batch_normalization(name='resx2_conv3_bn'))

        (self.feed('resx1_elewise_relu', 
                   'resx2_conv3_bn')
             .add(name='resx2_elewise')
             .relu(name='resx2_elewise_relu')
             .conv(5, 5, 64, 1, 1, name='conv3')
             .conv(1, 1, 1000, 1, 1, name='conv4')
             .conv(1, 1, 400, 1, 1, name='conv5')
             .conv(1, 1, 1, 1, 1, relu=False, name='conv6'))
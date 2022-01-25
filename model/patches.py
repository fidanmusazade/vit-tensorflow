import tensorflow as tf

class PatchCreator(tf.keras.layers.Layer):
    '''
    Used to create patches to be fed into ViT
    Input:
        patch_size - size of the single patch
        images - images to be divided into patches
    Output:
        patches - images divided into patches of given size
    '''
    def __init__(self, patch_size):
        super(PatchCreator, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


class PatchEncoder(tf.keras.layers.Layer):
    '''
    Encode patches and add positional encoding
    Input:
        num_patches - number of patches to encode
        projection_dim - projection dimension
    Output:
        encoded - encoded patches
    '''
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        
        self.dense_projection = tf.keras.layers.Dense(self.projection_dim)
        self.positions = tf.reshape(tf.range(start=0, limit=self.num_patches, delta=1), 
                                    (self.num_patches,))
        self.position_embedding = tf.keras.layers.Embedding(input_dim=self.num_patches, 
                                                            output_dim=self.projection_dim)

    def call(self, patch):
        encoded = self.dense_projection(patch) + self.position_embedding(self.positions)
        return encoded
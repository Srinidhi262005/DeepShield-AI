import numpy as np
try:
    import tensorflow as tf
except ImportError:
    tf = None

import cv2

class GradCAM:
    """
    Computes Grad-CAM heatmaps to provide explainability for model predictions.
    """
    def __init__(self, model, layer_name="block14_sepconv2_act"):
        """
        :param model: The trained Keras model
        :param layer_name: The name of the last convolutional layer in Xception
        """
        self.model = model
        self.layer_name = layer_name
        self.grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(layer_name).output, model.output]
        )

    def compute_heatmap(self, image_array, class_idx=0):
        """
        Generates the raw heatmap.
        """
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = self.grad_model(image_array)
            loss = preds[:, class_idx]

        # Gradients of the output class with respect to the output feature map
        grads = tape.gradient(loss, last_conv_layer_output)

        # Mean intensity of the gradients over each feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Multiply each channel in the feature map by 'how important this channel is'
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # For visualization, we normalize the heatmap between 0 & 1
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()

    def overlay_heatmap(self, heatmap, original_image, alpha=0.4, colormap=cv2.COLORMAP_JET):
        """
        Overlays the heatmap on the original image.
        """
        # Rescale heatmap to 0-255
        heatmap = np.uint8(255 * heatmap)

        # Use jet colormap to colorize heatmap
        jet = cv2.applyColorMap(heatmap, colormap)
        jet = cv2.resize(jet, (original_image.shape[1], original_image.shape[0]))

        # Superimpose the heatmap on original image
        superimposed_img = jet * alpha + original_image
        superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

        return superimposed_img

if __name__ == "__main__":
    print("GradCAM module loaded.")

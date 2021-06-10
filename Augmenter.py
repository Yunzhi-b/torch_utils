class ColorAugmenter:
    def __init__(self):
        # settings
        self.use_gamma = True
        self.use_color_jitter = True
        self.use_random_blur = True
        self.use_random_noise = True

        # switches
        self.adjust_gamma = False
        self.jitter_color = False
        self.random_blur = False
        self.random_noise = False

        # probs
        self.adjust_gamma_prob = 0.2
        self.color_jitter_prob = 0.2
        self.colorjitter_op = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0)
        self.random_blur_prob = 0.1
        self.random_noise_prob = 0.1

        # random params
        self.color_transform = None
        self.gamma_val = None
        self.f_size = None
        self.noise_mode = None
        self.seed = None
        self.var = None
        self.amount = None

    def fix_random(self):
        """
        fix random numbers to operate the same transform on image pairs
        """
        # switches
        self.adjust_gamma = self.use_gamma and random.random() < self.adjust_gamma_prob
        self.jitter_color = self.use_color_jitter and random.random() < self.color_jitter_prob
        self.random_blur = self.use_random_blur and random.random() < self.random_blur_prob
        self.random_noise = self.use_random_noise and random.random() < self.random_noise_prob

        # params
        self.gamma_val = np.random.uniform(0.8, 1.2, 1)
        self.color_transform = transforms.ColorJitter.get_params(self.colorjitter_op.brightness,
                                                                 self.colorjitter_op.contrast,
                                                                 self.colorjitter_op.saturation,
                                                                 self.colorjitter_op.hue)
        self.f_size = random.choice([3, 5])
        self.noise_mode = random.choice(['gaussian', 'pepper', 's&p', 'speckle'])
        self.seed = np.random.randint(2 ** 32 - 1)
        self.var = (random.randint(0, 16) / 255) ** 2
        self.amount = random.randint(0, 8) / 255

    def augment(self, img):
        """
        augment single image. please call fix_random() prior to augment a pair of images
        :param img: raw image, HWC, np.uint8
        :return: augmented image, HWC, np.uint8
        """
        # adjust gamma
        if self.adjust_gamma:
            tmp_img = 255. * ((img.copy() / 255.) ** self.gamma_val)
            # src_img = src_mask * tmp_src_img + (1 - src_mask) * src_img
            img = np.uint8(tmp_img)

        # color jitter
        if self.jitter_color:
            tmp_img = self.color_transform(Image.fromarray(img))
            # # src_img = src_mask * tmp_src_img + (1 - src_mask) * src_img
            img = np.uint8(tmp_img)

        # random blur
        if self.random_blur:
            tmp_img = cv2.boxFilter(img.copy(), -1, (self.f_size, self.f_size))
            # src_img = src_mask * tmp_src_img + (1 - src_mask) * src_img
            img = np.uint8(tmp_img)

        # random noise
        if self.random_noise:
            if self.noise_mode == 'gaussian' or self.noise_mode == 'speckle':
                tmp_img = skimage.img_as_ubyte(
                    skimage.util.random_noise(img.copy(), mode=self.noise_mode, mean=0, var=self.var, seed=self.seed))
                # src_img = src_mask * tmp_src_img + (1 - src_mask) * src_img
                img = np.uint8(tmp_img)
            else:
                tmp_img = skimage.img_as_ubyte(
                    skimage.util.random_noise(img.copy(), mode=self.noise_mode, amount=self.amount, seed=self.seed))
                # src_img = src_mask * tmp_src_img + (1 - src_mask) * src_img
                img = np.uint8(tmp_img)

        return img
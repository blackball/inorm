/**
 * Illumination normalization pipe line.
 *
 * @blackball
 */

#include "inorm.h"
#include <stdlib.h>
#include <malloc.h>
#include <string.h>

#include <opencv/cv.h>
#include <opencv/highgui.h>

#include <assert.h>

struct fmat_t {
        float *data;
        int w, h;
};

#define fmat_at(mm, i, j) ((mm)->data[(i) * (mm)->w + (j)])

static struct fmat_t*
fmat_new(int w, int h) {
        struct fmat_t *fm = malloc(sizeof(*fm));
        fm->data = malloc(sizeof(float) * w * h);
        fm->w = w;
        fm->h = h;
        return fm;
}

static void
fmat_free(struct fmat_t **fm) {
        if (fm && (*fm)) {
                free( (*fm)->data );
                free( *fm );
                *fm = NULL;
        }
}

static float
fmat_min(const struct fmat_t *fm) {
        const int sz = fm->w * fm->h;

        int i;

        const float *fdata = fm->data;
        
        float m =  fdata[0];
        
        for (i = 1; i < sz; ++i) {
                if (fdata[i] < m) {
                        m = fdata[i];
                }
        }
        return m;
}

static float
fmat_max(const struct fmat_t *fm) {
        const int sz = fm->w * fm->h;
        const float *fdata = fm->data;
        
        float m =  fdata[0];
        int i;
        
        for (i = 1; i < sz; ++i) {
                if (fdata[i] > m) {
                        m = fdata[i];
                }
        }
        return m;
}

static void
fmat_sub(const struct fmat_t *ma, const struct fmat_t *mb, struct fmat_t *dst) {
        const int size = ma->w * ma->h;
        int i = 0;
        for (; i < size; ++i) {
                dst->data[i] = ma->data[i] - mb->data[i];
        }
}

#define ROW_ALIGN(w) (((w) + 3) & ~3)

static struct image_t*
image_new(int w, int h) {
        const int ws = ROW_ALIGN(w);
        
        struct image_t *img = malloc(sizeof(*img));
        img->data = malloc(sizeof(unsigned char) * ws * h);
        img->w = w;
        img->h = h;
        img->ws = ws;
        
        return img;
}

static void
image_free(struct image_t **img) {
        if (img && (*img)) {
                free( (*img)->data );
                free( *img );
                *img = NULL;
        }
}

/* First we need do  *gamma correction* with gamma = 0.2
 * Then we need a *DoG*: sigma = 2.0, sigma = 1.0
 * Next, *rescale* to an image.
 * Finally, *meadian filtering*.
 */

static void
gamma_correction(const struct image_t *img, struct fmat_t *fm) {
        const int ws = img->ws;
        const int w  = img->w;
        const int h  = img->h;
        const unsigned char *idata = img->data;
        float *fdata = fm->data;

        /* here is a table of [0~255]^gamma */
        static const float _gamma_tab[] = {
                0.0f, 84.1845f, 96.7026f, 104.87124f, 111.08212f, 116.15185f, 120.46542f, 124.23723f, 127.59984f, 130.64135f, 133.42344f, 135.99116f, 138.37843f, 140.61149f, 142.7111f, 144.69396f, 146.57373f, 148.36174f, 150.0675f, 151.69905f, 153.26329f, 154.76616f, 156.21282f, 157.6078f, 158.95507f, 160.25816f, 161.52019f, 162.74397f, 163.932f, 165.08657f, 166.20971f, 167.30329f, 168.369f, 169.4084f, 170.42289f, 171.41379f, 172.38229f, 173.3295f, 174.25645f, 175.16409f, 176.05329f, 176.92488f, 177.77963f, 178.61825f, 179.44141f, 180.24974f, 181.04382f, 181.82421f, 182.59143f, 183.34597f, 184.08828f, 184.81881f, 185.53798f, 186.24616f, 186.94373f, 187.63104f, 188.30842f, 188.9762f, 189.63467f, 190.28412f, 190.92482f, 191.55704f, 192.18101f, 192.79699f, 193.4052f, 194.00585f, 194.59915f, 195.1853f, 195.76449f, 196.33691f, 196.90274f, 197.46213f, 198.01525f, 198.56226f, 199.10331f, 199.63855f, 200.1681f, 200.69211f, 201.2107f, 201.724f, 202.23212f, 202.73519f, 203.23332f, 203.72661f, 204.21517f, 204.6991f, 205.17849f, 205.65345f, 206.12406f, 206.5904f, 207.05258f, 207.51067f, 207.96474f, 208.41489f, 208.86117f, 209.30368f, 209.74248f, 210.17763f, 210.60921f, 211.03728f, 211.46191f, 211.88315f, 212.30107f, 212.71572f, 213.12717f, 213.53546f, 213.94065f, 214.3428f, 214.74195f, 215.13816f, 215.53147f, 215.92192f, 216.30958f, 216.69447f, 217.07665f, 217.45616f, 217.83303f, 218.20732f, 218.57906f, 218.94828f, 219.31503f, 219.67934f, 220.04125f, 220.4008f, 220.75802f, 221.11293f, 221.46559f, 221.81601f, 222.16423f, 222.51028f, 222.8542f, 223.196f, 223.53572f, 223.87339f, 224.20903f, 224.54268f, 224.87435f, 225.20408f, 225.53189f, 225.85781f, 226.18185f, 226.50405f, 226.82442f, 227.143f, 227.45979f, 227.77484f, 228.08815f, 228.39974f, 228.70965f, 229.01788f, 229.32447f, 229.62942f, 229.93277f, 230.23452f, 230.53469f, 230.83331f, 231.1304f, 231.42596f, 231.72002f, 232.0126f, 232.30371f, 232.59336f, 232.88158f, 233.16839f, 233.45378f, 233.73779f, 234.02042f, 234.3017f, 234.58163f, 234.86023f, 235.13752f, 235.4135f, 235.6882f, 235.96162f, 236.23378f, 236.50469f, 236.77436f, 237.04282f, 237.31006f, 237.5761f, 237.84096f, 238.10464f, 238.36716f, 238.62853f, 238.88876f, 239.14786f, 239.40584f, 239.66271f, 239.91849f, 240.17318f, 240.42679f, 240.67934f, 240.93084f, 241.18129f, 241.4307f, 241.67908f, 241.92645f, 242.17281f, 242.41818f, 242.66255f, 242.90594f, 243.14837f, 243.38982f, 243.63033f, 243.86989f, 244.10851f, 244.3462f, 244.58297f, 244.81883f, 245.05378f, 245.28783f, 245.52099f, 245.75328f, 245.98468f, 246.21522f, 246.4449f, 246.67373f, 246.90171f, 247.12885f, 247.35516f, 247.58064f, 247.8053f, 248.02916f, 248.25221f, 248.47446f, 248.69591f, 248.91658f, 249.13648f, 249.35559f, 249.57394f, 249.79153f, 250.00837f, 250.22445f, 250.43979f, 250.65439f, 250.86826f, 251.0814f, 251.29382f, 251.50553f, 251.71652f, 251.92681f, 252.1364f, 252.3453f, 252.5535f, 252.76103f, 252.96787f, 253.17404f, 253.37954f, 253.58437f, 253.78854f, 253.99206f, 254.19493f, 254.39716f, 254.59874f, 254.79969f, 255.0f
        };

        int i, j;
        for (i = 0; i < h; ++i) {
                for (j = 0; j < w; ++j) {
                        fdata[i * w + j] = _gamma_tab[ idata[i * ws + j] & 0xFF ];
                }
        }
}

static void 
fmat_print(const struct fmat_t *fm) {
        int i, j;
        printf("\n");
        for (i = 0; i < fm->h; ++i) {
                for (j = 0; j < fm->w; ++j) {
                        printf("%-3lf, ", fm->data[i * fm->w + j]);
                }
                printf("\n");
        }
}

static void 
border_replicate(const struct fmat_t *src, int rx, int ry, int rw, int rh, int border, struct fmat_t *dst) {
	/* @TODO safty check */
	int i, j;

	if ((dst->w != rw + border + border) || (dst->h != rh + border + border)) {
		assert(0);
	}

	/* copy main body */
	for (i = border; i < rh + border; ++i) {
		memcpy(&(fmat_at(dst, i, border)), &(fmat_at(src, ry + i - border, rx)), sizeof(float) * rw);
	}

	/* replicate first row */
	for (i = 0; i < border; ++i) {
		memcpy( &(fmat_at(dst, i, border)), &(fmat_at(dst, border, border)), sizeof(float) * rw);
	}

	/* replicate last row */
	for (i = rh + border; i < rh + border + border; ++i) {
		memcpy(&(fmat_at(dst, i, border)), &(fmat_at(dst, rh+border-1, border)), sizeof(float) * rw);
	}

	/* replicate left and right border */
	for (i = border; i < rh + border; ++i) {
		const float left_border = fmat_at(dst, i, border);
		const float right_border = fmat_at(dst, i, rw+border-1);

		for (j = 0; j < border; ++j) {
			fmat_at(dst, i, j) = left_border;
		}

		for (j = rh+border; j < rh+border+border; ++j) {
			fmat_at(dst, i, j) = right_border;
		}
	}
}

#define CAST_TO_8U(t) (unsigned char)(!((t) & ~255) ? (t) : (t) > 0 ? 255 : 0)

static void
rescale_to_image(const struct fmat_t *fm, struct image_t *dst) {
        /* after DoG, there're probably some negative values in fm */
        const float max = fmat_max(fm);
        const float min = fmat_min(fm);

        /* @TODO need protect from 0-div */
        const float dd = 255.f / (max - min);

        /* for a pixel value 255.0 * (v - min) / (max - min) */
        
        const int w = dst->w;
        const int h = dst->h;
        const int ws = dst->ws;
        const float *fdata = fm->data;
        unsigned char *idata = dst->data;
                
        int i, j;

        for (i = 0; i < h; ++i) {
                for (j = 0; j < w; ++j) {
                        unsigned int t = (unsigned int)((((float)fdata[i*w + j]) - min) * dd);
                        idata[i * ws + j] = CAST_TO_8U(t);
                }
        }
}

#define AT_ASSERT(i, j, w, h)  assert( (i >= 0 && i < h) && (j >= 0 && j < w) )

static void
sep_conv_h(const float *m, int w, int h, const float *kernel, int kernel_size, float *dst) {
        const int border = kernel_size / 2;
        const float *mdata = m;
        
        int i, j, k;

        if (kernel_size > w) {
                return ;
        }

        for (i = 0; i < h; ++i) {
                for (j = 0; j < border; ++j) {
                        float sum = 0.f;
                        int d = j - border;

                        for (k = 0; k < kernel_size; ++k, ++d) {
                                if (d < 0) {
                                        sum += kernel[k] * mdata[i * w];
                                }
                                else {
                                        sum += kernel[k] * mdata[i * w + d];
                                }
                        }
                        
                        dst[i * w + j] = sum;
                }
                
                for (j = border; j < w - border; ++j) {
                        float sum = 0.f;
                        int d = j - border;

                        for (k = 0; k < kernel_size; ++k, ++d) {
                                sum += kernel[k] * mdata[i * w + d];
                        }
                        
                        dst[i * w + j] = sum;
                }

                for (j = w - border; j < w; ++j) {
                        float sum = 0.f;
                        int d = j - border;

                        for (k = 0; k < kernel_size; ++k, ++d) {
                                if (d >= w) {
                                        sum += kernel[k] * mdata[i * w + w - 1];
                                }
                                else {
                                        sum += kernel[k] * mdata[i * w + d];
                                }
                        }

                        dst[i * w + j] = sum;
                }
        }
}

static void
sep_conv_v(const float *m, int w, int h, const float *kernel, int kernel_size, float *dst) {
        const int border = kernel_size / 2;
        const float *mdata = m;
        
        int i, j, k;

        if (kernel_size > h) {
                return ;
        }

        for (i = 0; i < border; ++i) {
                for (j = 0; j < w; ++j) {
                        float sum = 0.f;
                        int d = i - border;

                        for (k = 0; k < kernel_size; ++k, ++d) {
                                if (d < 0) {
                                        sum += kernel[k] * mdata[j];
                                }
                                else {
                                        sum += kernel[k] * mdata[d * w + j];
                                }
                        }

                        dst[i * w + j] = sum;
                }
        }

        for (i = border; i < h - border; ++i) {
                for (j = 0; j < w; ++j) {
                        float sum = 0.f;
                        int d = i - border;

                        for (k = 0; k < kernel_size; ++k, ++d) {
                                sum += kernel[k] * mdata[d * w + j];
                        }

                        dst[i * w + j] = sum;
                }
        }

        for (i = h - border; i < h; ++i) {
                for (j = 0; j < w; ++j) {
                        float sum = 0.f;
                        int d = i - border;

                        for (k = 0; k < kernel_size; ++k, ++d) {
                                if (d >= h) {
                                        sum += kernel[k] * mdata[w*h - w + j];
                                }
                                else {
                                        sum += kernel[k] * mdata[d * w + j];
                                }
                        }

                        dst[i * w + j] = sum;
                }
        }
}

static void 
seperate_conv(const struct fmat_t *fm, const float kernel[], unsigned int kernel_size, struct fmat_t *dst) {
	struct fmat_t *buf = fmat_new(fm->w, fm->h);
	sep_conv_h(fm->data, fm->w, fm->h, kernel, kernel_size, buf->data);
	sep_conv_v(buf->data, buf->w, buf->h, kernel, kernel_size, dst->data);
	fmat_free(&buf);
}

static void
difference_of_gaussian(struct fmat_t *fm) {
        /* this kernel was generated by the gen-gaussian-kernel.py script */
#define KERNEL_A_SIZE 7
#define KERNEL_B_SIZE 13
        /* sigma = 1.0 */
        static const float kernel_a[] = {
                0.00443f, 0.05401f, 0.24204f,
                0.39905f,
                0.24204f, 0.05401f, 0.00443f,
        };
        
        /* sigma = 2.0 */
        static const float kernel_b[] = {
                0.00222f, 0.00877f, 0.02702f, 0.06482f, 0.12111f, 0.17621f,
                0.19968f,
                0.17621f, 0.12111f, 0.06482f, 0.02702f, 0.00877f, 0.00222f,
        }; 

        /* gaussian smoothing */
        struct fmat_t *fma = fmat_new(fm->w, fm->h);
        struct fmat_t *fmb = fmat_new(fm->w, fm->h);

        seperate_conv(fm, kernel_a, KERNEL_A_SIZE, fma);
        seperate_conv(fm, kernel_b, KERNEL_B_SIZE, fmb);

        fmat_sub(fma, fmb, fm);

        fmat_free(&fma);
        fmat_free(&fmb);
}

#ifndef ABS
#define ABS(v) ((v) > 0 ? (v) : -(v))
#endif

static void
tanh_smooth(const struct fmat_t *src, struct fmat_t *dst) {
        const float trim = 10.f;
        const float trim_inv = 1.f / trim;
        float mean = 0.f;
        const float a = 0.1;
        const int size = src->w * src->h;
        
        int i, j;
        for (i = 0; i < size; ++i) {
                mean += powf(ABS(src->data[i]), a); 
        }

        mean /= size;

        mean = 1.0f / powf(mean, 1.f/a);

        for (i = 0; i < size; ++i) {
                dst->data[i] = src->data[i] * mean;
        }

        mean = 0.f;
        for (i = 0; i < size; ++i) {
                mean += powf(MIN(trim, ABS(dst->data[i])), a);
        }

        mean /= size;

        mean = 1.0f / powf(mean, 1.0f/a);

        for (i = 0; i < size; ++i) {
                dst->data[i] *= mean;
        }

        for (i = 0; i < size; ++i) {
                dst->data[i] = trim * tanh( dst->data[i] * trim_inv);
        }
}


void
inorm(const struct image_t *src, struct image_t *dst) {
        struct fmat_t *fm = fmat_new(src->w, src->h);
        struct fmat_t *tfm = fmat_new(src->w, src->h);
        struct image_t *timg = image_new(src->w, src->h);
        
        gamma_correction(src, fm);
        difference_of_gaussian(fm);
        tanh_smooth(fm, tfm);
        rescale_to_image(tfm, dst);
        
        image_free(&timg);
        fmat_free(&fm);
        fmat_free(&tfm);
}


#ifndef INORM_H
#define INORM_H

struct image_t {
        unsigned char *data;
        int w, ws, h;
};

void inorm(const struct image_t *src, struct image_t *dst);

#endif

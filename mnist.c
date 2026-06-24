#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

static uint32_t read_be32(FILE *f) {
    uint32_t x;
    fread(&x, 4, 1, f);
    // byte-swap: big -> little endian
    return ((x & 0xFF000000) >> 24) |
           ((x & 0x00FF0000) >>  8) |
           ((x & 0x0000FF00) <<  8) |
           ((x & 0x000000FF) << 24);
}

int main(void) {
    /* ---- labels ---- */
    FILE *lf = fopen("train-labels.idx1-ubyte", "rb");
    if (!lf) { perror("labels"); return 1; }

    uint32_t magic = read_be32(lf);
    if (magic != 0x00000801) { fprintf(stderr, "bad label magic: %08X\n", magic); return 1; }

    uint32_t n = read_be32(lf);
    uint8_t *labels = (uint8_t *)malloc(n);
    fread(labels, 1, n, lf);
    fclose(lf);

    /* ---- images ---- */
    FILE *imf = fopen("train-images.idx3-ubyte", "rb");
    if (!imf) { perror("images"); return 1; }

    magic = read_be32(imf);
    if (magic != 0x00000803) { fprintf(stderr, "bad image magic: %08X\n", magic); return 1; }

    uint32_t n_img = read_be32(imf);
    uint32_t rows  = read_be32(imf);
    uint32_t cols  = read_be32(imf);
    uint32_t sz    = rows * cols;          // 784 for MNIST

    uint8_t *images = (uint8_t *)malloc(n_img * sz);  // ~45MB for train set
    fread(images, 1, n_img * sz, imf);
    fclose(imf);

    /* ---- access pattern ----
       pixel (r,c) of image i:  images[i*sz + r*cols + c]
       label of image i:        labels[i]
    ---- */

    // sanity check
    if (n_img != n) fprintf(stderr, "warn: image count %u != label count %u\n", n_img, n);

    // ASCII-render first image
    printf("Label: %d\n", labels[0]);
    for (uint32_t r = 0; r < rows; r++) {
        for (uint32_t c = 0; c < cols; c++) {
            uint8_t px = images[r * cols + c];
            putchar(px > 128 ? '#' : px > 32 ? '.' : ' ');
        }
        putchar('\n');
    }

    free(images);
    free(labels);
    return 0;
}
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <uchar.h>

static uint32_t read_be32(FILE *f) {
    uint32_t x;
    fread(&x, 4, 1, f);
    // byte-swap: big -> little endian
    return ((x & 0xFF000000) >> 24) |
           ((x & 0x00FF0000) >>  8) |
           ((x & 0x0000FF00) <<  8) |
           ((x & 0x000000FF) << 24);
}


// ASCII-render nth image
void render_image(uint32_t rows, uint32_t cols, uint8_t *images, uint8_t *labels, uint32_t s) {
    printf("Label: %d\n", labels[s]);
    for (uint32_t r = 0; r < rows; r++) {
        for (uint32_t c = 0; c < cols; c++) {
            uint8_t px = images[rows*cols*s + r * cols + c];
            const char *outchar = (px > 191 ? "\u2588\u2588" :
                            px > 128 ? "\u2593\u2593" : 
                            px > 64 ? "\u2592\u2592" :
                            px > 32 ? "\u2591\u2591" : 
                            "  ");
            fputs(outchar, stdout);
        }
        putchar('\n');
    }
}

uint8_t *copy_image(uint32_t rows, uint32_t cols, uint8_t *images, uint32_t s) {
    uint8_t *dst = malloc(rows*cols);
    if (!dst) return NULL;

    for(uint32_t n = 0; n < rows*cols; n++) {
        dst[n] = images[rows*cols*s + n];
    }
    return dst;
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
    uint32_t sz    = rows * cols;          // 784 for MNIST, 28x28

    uint8_t *images = (uint8_t *)malloc(n_img * sz);  // ~45MB for train set
    fread(images, 1, n_img * sz, imf);
    fclose(imf);

    /* ---- access pattern ----
       pixel (r,c) of image i:  images[i*sz + r*cols + c]
       label of image i:        labels[i]
    ---- */

    // sanity check
    if (n_img != n) fprintf(stderr, "warn: image count %u != label count %u\n", n_img, n);

    uint32_t select;
    printf("Selected image (starts at 0): ");
    scanf("%u",&select);
    render_image(rows, cols, images, labels, select);

    uint8_t *single = copy_image(rows, cols, images, select);
    printf("test: %u\n", single[28*7 + 4]);
    
    free(images);
    free(labels);
    return 0;
}
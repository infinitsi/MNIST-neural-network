#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <uchar.h>
#include <math.h>
#include <time.h>

static uint32_t read_be32(FILE *f) {
    uint32_t x;
    fread(&x, 4, 1, f);
    // byte-swap: big -> little endian
    return ((x & 0xFF000000) >> 24) |
           ((x & 0x00FF0000) >>  8) |
           ((x & 0x0000FF00) <<  8) |
           ((x & 0x000000FF) << 24);
}

typedef struct {
    uint8_t *images;
    uint8_t *labels;
    uint32_t count;
    uint32_t rows;
    uint32_t cols;
} MnistDataset;

void render_image(const MnistDataset *ds, uint32_t s) {
    printf("Label: %d\n", ds->labels[s]);
    uint32_t sz = ds->rows * ds->cols;
    for (uint32_t r = 0; r < ds->rows; r++) {
        for (uint32_t c = 0; c < ds->cols; c++) {
            uint8_t px = ds->images[sz*s + r * ds->cols + c];
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

uint8_t *copy_image(const MnistDataset *ds, uint32_t s) {
    uint32_t sz = ds->rows * ds->cols;
    uint8_t *dst = malloc(sz);
    if (!dst) return NULL;

    for(uint32_t n = 0; n < sz; n++) {
        dst[n] = ds->images[sz*s + n];
    }
    return dst;
}

typedef struct {
    uint32_t inputct;
    uint32_t outputct;
    float *weights; // length = inputct*outputct
    float *biases;  // length = outputct
} Layer;

void init_layer(Layer *l, uint32_t inct, uint32_t outct) {
    l->inputct = inct;
    l->outputct = outct;
    l->weights = calloc(inct*outct, sizeof(float));
    l->biases = calloc(outct, sizeof(float));
}

void randomize_weights_biases(Layer *l) {
    if(!l || !l->weights || !l->biases) return;
    srand(time(NULL));
    for(uint32_t i = 0; i < (l->inputct * l->outputct)/sizeof(float); ++i) {
        l->weights[i] = ((float)rand() / RAND_MAX) * 0.1f - 0.05f;
    }
    // for (uint32_t i = 0; i < l->outputct; ++i) {
    //     l->biases[i] = 0.0f;
    // }
}

void reset_file(const char *path) {
    FILE *p = fopen(path, "w");
    if(!p) {perror("path");}
    fclose(p);
}

void write_layer_to_file(const char *layer_path, const Layer *l) {
    FILE *lp = fopen(layer_path, "wb");
    if(!lp) {perror("layer path");}
    fwrite(&l->inputct, sizeof(uint32_t), 1, lp);
    fwrite(&l->outputct, sizeof(uint32_t), 1, lp);
    fwrite(l->weights, sizeof(float), l->inputct * l->outputct, lp);
    fwrite(l->biases, sizeof(float), l->outputct, lp);
    fclose(lp);
}

void read_file_to_layer(const char *layer_path, Layer *l) {
    FILE *lp = fopen(layer_path, "rb");
    if(!lp) {perror("layer path");}
    uint32_t inct;
    uint32_t outct;
    fread(&inct, sizeof(uint32_t), 1, lp);
    fread(&outct, sizeof(uint32_t), 1, lp);
    init_layer(l, inct, outct);
    fread(l->weights, sizeof(float), l->inputct * l->outputct, lp);
    fread(l->biases, sizeof(float), l->outputct, lp);
    fclose(lp);
}

int main(void) {
    MnistDataset dataSet;
    /* ---- labels ---- */
    FILE *lf = fopen("train-labels.idx1-ubyte", "rb");
    if (!lf) { perror("labels"); return 1; }

    uint32_t magic = read_be32(lf);
    if (magic != 0x00000801) { fprintf(stderr, "bad label magic: %08X\n", magic); return 1; }

    uint32_t n = read_be32(lf);
    dataSet.labels = (uint8_t *)malloc(n);
    fread(dataSet.labels, 1, n, lf);
    fclose(lf);

    /* ---- images ---- */
    FILE *imf = fopen("train-images.idx3-ubyte", "rb");
    if (!imf) { perror("images"); return 1; }

    magic = read_be32(imf);
    if (magic != 0x00000803) { fprintf(stderr, "bad image magic: %08X\n", magic); return 1; }

    dataSet.count = read_be32(imf);
    dataSet.rows  = read_be32(imf);
    dataSet.cols  = read_be32(imf);
    uint32_t sz    = dataSet.rows * dataSet.cols;          // 784 for MNIST, 28x28

    dataSet.images = (uint8_t *)malloc(dataSet.count * sz);  // ~45MB for train set
    fread(dataSet.images, 1, dataSet.count * sz, imf);
    fclose(imf);

    /* ---- access pattern ----
       pixel (r,c) of image i:  images[i*sz + r*cols + c]
       label of image i:        labels[i]
    ---- */
    // sanity check
    if (dataSet.count != n) fprintf(stderr, "warn: image count %u != label count %u\n", dataSet.count, n);

    uint32_t select = -1;
    while(select < 0 || select >= dataSet.count) {
        printf("Selected image (0-59999): ");
        scanf("%u",&select);
    }
    render_image(&dataSet, select);

    uint8_t *single = copy_image(&dataSet, select);
    printf("test: %u\n", single[28*7 + 4]);

    char layerpath[] = "layerfile";
    Layer *layer1 = malloc(sizeof *layer1);
    // init_layer(layer1, sz, 6);
    // randomize_weights_biases(layer1);
    read_file_to_layer(layerpath, layer1);
    write_layer_to_file(layerpath, layer1);



    free(dataSet.images);
    free(dataSet.labels);
    free(layer1->weights);
    free(layer1->biases);
    free(layer1);
    return 0;
}
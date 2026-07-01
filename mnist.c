#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <uchar.h>
#include <math.h>
#include <time.h>

uint32_t read_be32(FILE *f) {
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
    /* ---- access pattern ----
       pixel (r,c) of image i:  images[i*sz + r*cols + c]
       label of image i:        labels[i]
    ---- */
    printf("Label: %d\n", ds->labels[s]);
    uint32_t sz = ds->rows * ds->cols;
    for (uint32_t r = 0; r < ds->rows; r++) {
        for (uint32_t c = 0; c < ds->cols; c++) {
            uint8_t px = ds->images[sz*s + r * ds->cols + c];
            //unicode shading
            //
            //░░▒▒▓▓██
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

void copy_image(float *dst, const MnistDataset *ds, uint32_t s) {
    uint32_t sz = ds->rows * ds->cols;
    for(uint32_t n = 0; n < sz; n++) {
        dst[n] = (float)(ds->images[sz*s + n]/255.f);
    }
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

void randomize_params(Layer *l) {
    if(!l || !l->weights || !l->biases) return;
    for(uint32_t i = 0; i < l->inputct * l->outputct; ++i) {
        l->weights[i] = ((float)rand() / RAND_MAX) * 0.3f - 0.15f;
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

void write_layer_to_file(const char *layer_path, const Layer *l) {
    FILE *lp = fopen(layer_path, "wb");
    if(!lp) {perror("layer path");}
    fwrite(&l->inputct, sizeof(uint32_t), 1, lp);
    fwrite(&l->outputct, sizeof(uint32_t), 1, lp);
    fwrite(l->weights, sizeof(float), l->inputct * l->outputct, lp);
    fwrite(l->biases, sizeof(float), l->outputct, lp);
    fclose(lp);
}

float sigmoid(float x) {
    return (1.f / (1.f + expf(-x)));
}

float reLU(float x) {
    return fmax(0.0,x);
}

//write weights*inputs+bias to neuron layer
void calc_neuronlayer(float *neurons, Layer *l, float *inp) {
    for(uint32_t i = 0; i < l->outputct; ++i) {
        float sum = 0.0f;
        for(uint32_t j = 0; j < l->inputct; ++j) {
            sum += l->weights[i*l->inputct + j] * inp[j];
        }
        neurons[i] = sigmoid(sum + l->biases[i]);
    }
}

// float calc_cost(uint8_t label, float *outputs, uint32_t outputct) {
//     float sum = 0.0f;
//     for(uint32_t i = 0; i < outputct; ++i) {
//         if(i == label-1) {
//             sum += (outputs[i]-1)*(outputs[i]-1);
//         } else {
//             sum += outputs[i]*outputs[i];
//         }
//     }
//     return sum;
// }

float calc_cost(uint8_t label, float *neuron0, Layer *l1, Layer *l2) {
    //neuron0 (input) --l1--> neuron1 --l2--> neuron2 (output) --label--> cost
    float neuron1[l1->outputct];
    calc_neuronlayer(neuron1, l1, neuron0);
    float neuron2[l2->outputct];
    calc_neuronlayer(neuron2, l2, neuron1);

    float sum = 0.0f;
    for(uint32_t i = 0; i < l2->outputct; ++i) {
        if(i == label-1) {
            sum += (neuron2[i]-1)*(neuron2[i]-1);
        } else {
            sum += neuron2[i]*neuron2[i];
        }
    }
    return sum;
}

int main(void) {
    MnistDataset dataSet;
    //labels (heap)
    FILE *lf = fopen("train-labels.idx1-ubyte", "rb");
    if (!lf) { perror("labels"); return 1; }

    uint32_t magic = read_be32(lf);
    if (magic != 0x00000801) { fprintf(stderr, "bad label magic: %08X\n", magic); return 1; }

    uint32_t n = read_be32(lf);
    dataSet.labels = (uint8_t *)malloc(n);
    fread(dataSet.labels, 1, n, lf);
    fclose(lf);

    //images (heap)
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

    // sanity check
    if (dataSet.count != n) fprintf(stderr, "warn: image count %u != label count %u\n", dataSet.count, n);

    //test render
    uint32_t select = 60000;
    while(select >= dataSet.count) {
        printf("Selected image (0-59999): ");
        scanf("%u",&select);
    }
    render_image(&dataSet, select);





    /* neural network */

    //image label
    uint8_t inputlabel = dataSet.labels[select];
    //neuron layer 0 - input (heap)
    float *neuron0 = malloc(sz * sizeof *neuron0);
    copy_image(neuron0, &dataSet, select);
    // printf("test: %u\n", neuron0[28*7 + 4]);

    //connection layer 1 (heap)
    char layer1path[] = "layer1data.bin";
    Layer *layer1 = malloc(sizeof *layer1);
    if (!layer1) { perror("layer1"); }
    //randomize parameters
    // init_layer(layer1, sz, 6);
    // srand((unsigned)time(NULL));
    // randomize_params(layer1);
    //read parameters from file
    read_file_to_layer(layer1path, layer1);

    //connection layer 2 (heap)
    char layer2path[] = "layer2data.bin";
    Layer *layer2 = malloc(sizeof *layer2);
    if (!layer2) { perror("layer2"); }
    //randomize parameters
    // srand((unsigned)time(NULL));
    // init_layer(layer2, layer1->outputct, 10);
    // randomize_params(layer2);
    //read parameters from file
    read_file_to_layer(layer2path, layer2);

    printf("Cost: %f\n", calc_cost(inputlabel, neuron0, layer1, layer2));




    //save
    write_layer_to_file(layer1path, layer1);
    write_layer_to_file(layer2path, layer2);

    free(dataSet.labels);
    free(dataSet.images);

    free(neuron0);
    free(layer1->weights);
    free(layer1->biases);
    free(layer1);
    free(layer2->weights);
    free(layer2->biases);
    free(layer2);
    return 0;
}
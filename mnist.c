#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <uchar.h>
#include <math.h>
#include <time.h>
#include <string.h>

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

float sigmoid(float x) { return (1.f / (1.f + expf(-x))); }

float dsigmoid(float x) { float s = sigmoid(x); return s * (1.0f - s); }

float reLU(float x) { return fmax(0.0,x); }

float dreLU(float x) { return x > 0.0f ? 1.0f : 0.0f; }

//write weights*inputs+bias to neuron layer
void calc_neurons(float *neurons, float *preactivation_neurons, Layer *l, float *inp) {
    for(uint32_t i = 0; i < l->outputct; ++i) {
        float sum = 0.0f;
        for(uint32_t j = 0; j < l->inputct; ++j) {
            sum += l->weights[i*l->inputct + j] * inp[j];
        }
        if(!preactivation_neurons) {
            neurons[i] = reLU(sum + l->biases[i]);
        } else {
            preactivation_neurons[i] = sum + l->biases[i];
            neurons[i] = reLU(preactivation_neurons[i]);
        }
    }
}

float calc_cost(uint8_t label, float *neuron0, Layer *l1, Layer *l2) {
    //neuron0 (input) --l1--> neuron1 --l2--> neuron2 (output) --label--> cost
    float neuron1[l1->outputct];
    calc_neurons(neuron1, NULL, l1, neuron0);
    float neuron2[l2->outputct];
    calc_neurons(neuron2, NULL, l2, neuron1);

    float sum = 0.0f;
    for(uint32_t i = 0; i < l2->outputct; ++i) {
        if(i == label) {
            sum += (neuron2[i]-1)*(neuron2[i]-1);
        } else {
            sum += neuron2[i]*neuron2[i];
        }
    }
    return sum;
}

void calc_gradient(Layer *dl1, Layer *dl2, float *neuron0, Layer *l1, Layer *l2, uint8_t label) {
    //
    float pre1[l1->outputct];
    float neuron1[l1->outputct];
    calc_neurons(neuron1, pre1, l1, neuron0);
    float pre2[l2->outputct];
    float neuron2[l2->outputct];
    calc_neurons(neuron2, pre2, l2, neuron1);
    //dc/da_j(L) or dcost/dneuron2
    float dn2[l2->outputct];
    float y;
    for(uint32_t i = 0; i < l2->outputct; ++i) {
        y = (i == label ? 1.0f : 0.0f);
        dn2[i] = 2.0f * (neuron2[i] - y);
    }
    //dc/db_j(L) or dcost/dbias2
    for(uint32_t i = 0; i < l2->outputct; ++i) {
        dl2->biases[i] = dreLU(pre2[i]) * dn2[i];
    }
    //dc/dw_jk(L) or dcost/dweights2
    //j for neuron2, k for neuron1
    for(uint32_t j = 0; j < l2->outputct; ++j) {
        for(uint32_t k = 0; k < l2->inputct; ++k) {
            dl2->weights[j*l2->inputct + k] = neuron1[k] * dl2->biases[j];
        }
    }
    //dc/da_k(L-1) or dcost/dneuron1
    float dn1[l2->inputct];
    for(uint32_t k = 0; k < l2->inputct; ++k) {
        float sum = 0;
        for(uint32_t j = 0; j < l2->outputct; ++j) {
            sum += l2->weights[j*l2->inputct + k] * dreLU(pre2[j]) * dn2[j];
        }
        dn1[k] = sum;
    }
    //dc/db_k(L-1) or dcost/dbias1
    for(uint32_t i = 0; i < l1->outputct; ++i) {
        dl1->biases[i] =  dreLU(pre1[i]) * dn1[i];
    }
    //dc/dw_kh(L-1) or dcost/dweights1
    //k for neuron1, h for neuron0
    for(uint32_t k = 0; k < l1->outputct; ++k) {
        for(uint32_t h = 0; h < l1->inputct; ++h) {
            dl1->weights[k*l1->inputct + h] = neuron0[h] * dl1->biases[k];
        }
    }
}


void add_gradient(Layer *dl1, Layer *dl2, Layer *dl1t, Layer *dl2t) {
    for (uint32_t j = 0; j < dl1->outputct; ++j) {
        for(uint32_t i = 0; i < dl1->inputct; ++i) {
            dl1->weights[j*dl1->inputct + i] += dl1t->weights[j*dl1->inputct + i];
        }
        dl1->biases[j]  += dl1t->biases[j];
    }
    for (uint32_t j = 0; j < dl2->outputct; ++j) {
        for(uint32_t i = 0; i < dl2->inputct; ++i) {
            dl2->weights[j*dl2->inputct + i] += dl2t->weights[j*dl2->inputct + i];
        }
        dl2->biases[j]  += dl2t->biases[j];
    }
}

void update_params(Layer *l1, Layer *l2, Layer *dl1, Layer *dl2, float scale) {
    for (uint32_t j = 0; j < l1->outputct; ++j) {
        for(uint32_t i = 0; i < l1->inputct; ++i) {
            l1->weights[j*l1->inputct + i] -= scale * dl1->weights[j*l1->inputct + i];
        }
        l1->biases[j]  -= scale * dl1->biases[j];
    }
    for (uint32_t j = 0; j < l2->outputct; ++j) {
        for(uint32_t i = 0; i < l2->inputct; ++i) {
            l2->weights[j*l2->inputct + i] -= scale * dl2->weights[j*l2->inputct + i];
        }
        l2->biases[j]  -= scale * dl2->biases[j];
    }
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
    // uint32_t select = 60000;
    // while(select >= dataSet.count) {
    //     printf("Selected image (0-59999): ");
    //     scanf("%u",&select);
    // }
    // render_image(&dataSet, select);





    /* neural network */

    //image label
    // uint8_t inputlabel = dataSet.labels[select];
    //neuron layer 0 - input (heap)
    // float *neuron0 = malloc(sz * sizeof *neuron0);
    // copy_image(neuron0, &dataSet, select);

    //connection layer 1 (heap)
    char layer1path[] = "layer1data.bin";
    Layer layer1;
    //read parameters from file
    read_file_to_layer(layer1path, &layer1);

    //connection layer 2 (heap)
    char layer2path[] = "layer2data.bin";
    Layer layer2;
    //read parameters from file
    read_file_to_layer(layer2path, &layer2);

    //randomize parameters
    // srand((unsigned)time(NULL));
    // init_layer(&layer1, sz, 6);
    // init_layer(&layer2, layer1.outputct, 10);
    // randomize_params(&layer1);
    // randomize_params(&layer2);


    /*  gradient descent */
    uint32_t batchsize = 10;
    uint32_t batches = 6000;
    float lr = 0.1f; //learning rate
    float avg_cost;
    Layer layer1gradient;
    Layer layer2gradient;
    init_layer(&layer1gradient, layer1.inputct, layer1.outputct);
    init_layer(&layer2gradient, layer2.inputct, layer2.outputct);
    //add temp to gradient batchsize times
    Layer temp1;
    Layer temp2;
    init_layer(&temp1, layer1.inputct, layer1.outputct);
    init_layer(&temp2, layer2.inputct, layer2.outputct);
    //input neuron0
    uint8_t inputlabel;
    float *neuron0 = malloc(sz * sizeof *neuron0);
    printf("Running...\n");
    for(uint32_t batchnum = 0; batchnum < batches; ++batchnum) {
        avg_cost = 0.0f;
        //reset gradient
        memset(layer1gradient.weights, 0, layer1.inputct*layer1.outputct*sizeof(float));
        memset(layer1gradient.biases,  0, layer1.outputct*sizeof(float));
        memset(layer2gradient.weights, 0, layer2.inputct*layer2.outputct*sizeof(float));
        memset(layer2gradient.biases,  0, layer2.outputct*sizeof(float));
        for(uint32_t select = batchnum*batchsize; select < (batchnum+1)*batchsize; ++select) {
            //input setup
            inputlabel = dataSet.labels[select];
            //printf("Label: %u\n", inputlabel); //show labels
            copy_image(neuron0, &dataSet, select);
            //cost and gradient
            avg_cost += calc_cost(inputlabel, neuron0, &layer1, &layer2);
            calc_gradient(&temp1, &temp2, neuron0, &layer1, &layer2, inputlabel);
            //test gradient
            // printf("w: %f, ", layer1.weights[408]);
            // printf("dc/dw: %f\n", temp1.weights[408]);
            //add to total gradient
            add_gradient(&layer1gradient, &layer2gradient, &temp1, &temp2);
        }
        //average cost
        avg_cost /= batchsize;
        // printf("Average cost: %f\n", avg_cost);
        //update layers
        update_params(&layer1, &layer2, &layer1gradient, &layer2gradient, lr/batchsize);
    }
    printf("Average cost: %f\n", avg_cost);

    //save
    write_layer_to_file(layer1path, &layer1);
    write_layer_to_file(layer2path, &layer2);

    free(dataSet.labels);
    free(dataSet.images);

    free(neuron0);
    free(layer1.weights);
    free(layer1.biases);
    free(layer2.weights);
    free(layer2.biases);
    free(layer1gradient.weights);
    free(layer1gradient.biases);
    free(layer2gradient.weights);
    free(layer2gradient.biases);
    free(temp1.weights);
    free(temp1.biases);
    free(temp2.weights);
    free(temp2.biases);
    return 0;
}
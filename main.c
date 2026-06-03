#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

int read_next_token(FILE *fp, char *buffer, int buffer_size) {
    int c;
    int i = 0;

    while ((c = fgetc(fp)) != EOF) {
        if (c == '#') {
            while ((c = fgetc(fp)) != EOF && c != '\n') {
            }
        } else if (c != ' ' && c != '\n' && c != '\t' && c != '\r') {
            break;
        }
    }

    if (c == EOF) {
        return 0;
    }

    do {
        if (i < buffer_size - 1) {
            buffer[i] = c;
            i++;
        }
        c = fgetc(fp);
    } while (c != EOF && c != ' ' && c != '\n' && c != '\t' && c != '\r');

    buffer[i] = '\0';
    return 1;
}

int main(int argc, char *argv[]) {
    omp_set_num_threads(40);
    if (argc != 2) { //check arg count
        fprintf(stderr, "Incorrect Args\n");
        exit(EXIT_FAILURE);
    }

    FILE *fp = fopen(argv[1], "r"); //open file in read
    if (fp == NULL) {
        fprintf(stderr, "Error opening image");
        exit(EXIT_FAILURE);
    }

    char buffer[100]; //curr line buffer
    read_next_token(fp, buffer, sizeof(buffer)); //scan file type, make sure P2 for grayscale
    if (strcmp(buffer, "P2") != 0) { 
        fprintf(stderr, "Incorrect file type");
        exit(EXIT_FAILURE);
    }
    //printf( "%s", buffer);

    unsigned long width, height; 
    read_next_token(fp, buffer, sizeof(buffer)); //scan next value, width
    sscanf(buffer, "%lu", &width);
    read_next_token(fp, buffer, sizeof(buffer)); //scan next value, height
    sscanf(buffer, "%lu", &height);
    unsigned long wh = width * height;

    int max_val;
    read_next_token(fp, buffer, sizeof(buffer)); //scans max pixel val
    sscanf(buffer, "%u", &max_val);

    //create dynamic arrays
    double *old = malloc(wh * sizeof(double));
    double *new = malloc(wh * sizeof(double));

    
    for (unsigned long i = 0; i < wh; i++) { //store pixels inside the old array
        unsigned long cur_pix;
        read_next_token(fp, buffer, sizeof(buffer));
        sscanf(buffer, "%lu", &cur_pix);
        old[i] = cur_pix;
    }
    unsigned long iterations = 200; //number of iterations for image smoothing

    double s_time = omp_get_wtime(); //start timing
    //Diffusion Iteration Loop
    float alpha = .2; //how much diffusion happens inside each step
    for (unsigned long step = 0; step < iterations; step++) {
        #pragma omp parallel for collapse(2) //parallelization loop for iterations over height and width
        for (unsigned long i = 0; i < height; i++) {
            for (unsigned long j = 0; j < width; j++) {
                unsigned long idx = i * width + j; //compute unique index
                //reflective boundary handling
                unsigned long top;
                unsigned long bottom;
                unsigned long left;
                unsigned long right;

                if (i == 0) {
                    top = idx;
                } else {
                    top = (i - 1) * width + j;
                }

                if (i == height - 1) {
                    bottom = idx;
                } else {
                    bottom = (i + 1) * width + j;
                }

                if (j == 0) {
                    left = idx;
                } else {
                    left = i * width + (j - 1);
                }

                if (j == width - 1) {
                    right = idx;
                } else {
                    right = i * width + (j + 1);
                }
                new[idx] = old[idx] + alpha * (old[top] + old[left] + old[right] + old[bottom] - 4 * old[idx]); //discrete approximation
            }
        }

        double *temp = old;
        old = new;
        new = temp;
    }

    double e_time = omp_get_wtime();
    printf("Smoothing time: %f seconds\n", e_time - s_time);

    FILE *out = fopen("output.pgm", "w");
    if (out == NULL) {
        fprintf(stderr, "Error creating output file\n");
        exit(EXIT_FAILURE);
    }

    // Write PGM header
    fprintf(out, "P2\n");
    fprintf(out, "%lu %lu\n", width, height);
    fprintf(out, "%u\n", max_val);

    // Write pixel values
    for (unsigned long i = 0; i < height; i++) {
        for (unsigned long j = 0; j < width; j++) {
            unsigned long idx = i * width + j;

            int pixel = (int) old[idx];

            // Clamp pixel values to valid range
            if (pixel < 0) {
                pixel = 0;
            }
            if (pixel > max_val) {
                pixel = max_val;
            }

            fprintf(out, "%d ", pixel);
        }
        fprintf(out, "\n");
    }

    fclose(out);



    return 0;
}

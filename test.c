#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

#define BUFFER_SIZE 1024
#define MAX_FILE_SIZE 50000

int main() {
    FILE *fptr;
    char guessList[MAX_FILE_SIZE];
    char lineBuffer[BUFFER_SIZE];
    fptr = fopen("guesslog","r");
    
    while(fgets(lineBuffer, BUFFER_SIZE, fptr)) {
        if(strlen(guessList) + strlen(lineBuffer) < MAX_FILE_SIZE-1) {
            strcat(guessList,lineBuffer);
        } else {
            printf("Storage buffer full, truncating\n");
            break;
        }
    }
    printf("%s",guessList);

    time_t currTime;
    time(&currTime);
    srand(currTime);
    int num = rand() % 51;
    int input;
    int guesses = 0;

    do {
        printf("Enter a number 0-50: ");
        scanf("%d", &input);
        guesses++;
        if(num > input) {
            printf("Greater\n");
        }
        if(num < input) {
            printf("Less\n");
        }
    } while(input != num);
    printf("You guessed it in %d guesses! It was %d", guesses, num);

    fptr = fopen("guesslog","a");
    fprintf(fptr, "%d guesses for %d\n", guesses, num);
    fclose(fptr);

    return 0;
}
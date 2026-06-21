#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {
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
    printf("You guessed it! It was %d", num);
    return 0;
}
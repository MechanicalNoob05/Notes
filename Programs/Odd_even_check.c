#include <stdlib.h>
#include <stdio.h>

int main()
{
    int numberToCheck;
    printf("Enter the number to Check: ");
    scanf("%d",&numberToCheck);

    if(numberToCheck%2== 0){
        printf("It is an even number\n");
    }else {
        printf("It is an odd number\n");
    }

    return 0;
}

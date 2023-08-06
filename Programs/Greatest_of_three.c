#include <stdio.h>

int main(){
    int one,two,three;
    printf("Enter first number: \n");
    scanf("%d",&one);
    printf("Enter second number: \n");
    scanf("%d",&two);
    printf("Enter third number: \n");
    scanf("%d",&three);

    if(one>two && one>three){
        printf("%d is greatest no of all\n",one);
    }else if (two>one && two > three) {
        printf("%d is greatest of of all\n",two);
    }else{
        printf("%d is greatest no of all\n",three);
    }

    return 0;
}

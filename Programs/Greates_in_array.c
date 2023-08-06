#include <stdio.h>

int main(){
    int n,largest;
    int arr[100];
    printf("Enter no of elements in range of 1 - 100\n");
    scanf("%d",&n);

    for(int i = 0; i<n;i++){
        printf("Enter %d number\n",i+1);
        scanf("%d",&arr[i]);
    }

    largest = arr[0];
    for(int i =1;i<n;i++){
        if(largest < arr[i]){
            largest = arr[i];
        }
    }
    printf("Lagets number is %d\n",largest);

    return 0;
}

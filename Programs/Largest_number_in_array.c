#include <stdio.h>

int returnMax(int array[],int n){
    int max = 0; 
    for ( int i=0;i<n;i++){
        if(array[i]>max){
            max = array[i];
        }
    }
    return max;
}
int main()
{
    int arr[]={1,40,12,90,198,4,7,24,660,120,88};
    int length = sizeof(arr) / sizeof(arr[0]);
    int max = returnMax(arr,length);
    printf("%d\n",max);
    return 0;
}


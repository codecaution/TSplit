    // Insert the begin and end event.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventRecord(start,0);

    float elapsedTime;
    cudaEventCreate(&stop);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start,stop);
    p->time = elapsedTime;
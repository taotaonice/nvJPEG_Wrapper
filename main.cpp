
#include <cuda_runtime_api.h>
#include "helper_nvJPEG.hxx"
#include <opencv2/opencv.hpp>

int dev_malloc(void **p, size_t s) { return (int)cudaMalloc(p, s); }

int dev_free(void *p) { return (int)cudaFree(p); }

typedef std::vector<std::string> FileNames;
typedef std::vector<std::vector<char> > FileData;

struct decode_params_t {
  std::string input_dir;
  int batch_size;
  int total_images;
  int dev;
  int warmup;

  nvjpegJpegState_t nvjpeg_state;
  nvjpegHandle_t nvjpeg_handle;
  cudaStream_t stream;

  nvjpegOutputFormat_t fmt;
  bool write_decoded;
  std::string output_dir;

  bool pipelined;
  bool batched;
};

int read_next_batch(FileNames &image_names, int batch_size,
                    FileNames::iterator &cur_iter, FileData &raw_data,
                    std::vector<size_t> &raw_len, FileNames &current_names) {
  int counter = 0;

  while (counter < batch_size) {
    if (cur_iter == image_names.end()) {
      std::cerr << "Image list is too short to fill the batch, adding files "
                   "from the beginning of the image list"
                << std::endl;
      cur_iter = image_names.begin();
    }

    if (image_names.size() == 0) {
      std::cerr << "No valid images left in the input list, exit" << std::endl;
      return EXIT_FAILURE;
    }

    // Read an image from disk.
    std::ifstream input(cur_iter->c_str(),
                        std::ios::in | std::ios::binary | std::ios::ate);
    if (!(input.is_open())) {
      std::cerr << "Cannot open image: " << *cur_iter
                << ", removing it from image list" << std::endl;
      image_names.erase(cur_iter);
      continue;
    }

    // Get the size
    std::streamsize file_size = input.tellg();
    input.seekg(0, std::ios::beg);
    // resize if buffer is too small
    if (raw_data[counter].size() < file_size) {
      raw_data[counter].resize(file_size);
    }
    if (!input.read(raw_data[counter].data(), file_size)) {
      std::cerr << "Cannot read from file: " << *cur_iter
                << ", removing it from image list" << std::endl;
      image_names.erase(cur_iter);
      continue;
    }
    raw_len[counter] = file_size;

    current_names[counter] = *cur_iter;

    counter++;
    cur_iter++;
  }
  return EXIT_SUCCESS;
}

// prepare buffers for RGBi output format
int prepare_buffers(FileData &file_data, std::vector<size_t> &file_len,
                    std::vector<int> &img_width, std::vector<int> &img_height,
                    std::vector<nvjpegImage_t> &ibuf,
                    std::vector<nvjpegImage_t> &isz, FileNames &current_names,
                    decode_params_t &params) {
  int widths[NVJPEG_MAX_COMPONENT];
  int heights[NVJPEG_MAX_COMPONENT];
  int channels;
  nvjpegChromaSubsampling_t subsampling;

  for (int i = 0; i < file_data.size(); i++) {
    checkCudaErrors(nvjpegGetImageInfo(
        params.nvjpeg_handle, (unsigned char *)file_data[i].data(), file_len[i],
        &channels, &subsampling, widths, heights));

    img_width[i] = widths[0];
    img_height[i] = heights[0];

    std::cout << "Processing: " << current_names[i] << std::endl;
    std::cout << "Image is " << channels << " channels." << std::endl;
    for (int c = 0; c < channels; c++) {
      std::cout << "Channel #" << c << " size: " << widths[c] << " x "
                << heights[c] << std::endl;
    }

    switch (subsampling) {
      case NVJPEG_CSS_444:
        std::cout << "YUV 4:4:4 chroma subsampling" << std::endl;
        break;
      case NVJPEG_CSS_440:
        std::cout << "YUV 4:4:0 chroma subsampling" << std::endl;
        break;
      case NVJPEG_CSS_422:
        std::cout << "YUV 4:2:2 chroma subsampling" << std::endl;
        break;
      case NVJPEG_CSS_420:
        std::cout << "YUV 4:2:0 chroma subsampling" << std::endl;
        break;
      case NVJPEG_CSS_411:
        std::cout << "YUV 4:1:1 chroma subsampling" << std::endl;
        break;
      case NVJPEG_CSS_410:
        std::cout << "YUV 4:1:0 chroma subsampling" << std::endl;
        break;
      case NVJPEG_CSS_GRAY:
        std::cout << "Grayscale JPEG " << std::endl;
        break;
      case NVJPEG_CSS_UNKNOWN:
        std::cout << "Unknown chroma subsampling" << std::endl;
        return EXIT_FAILURE;
    }

    int mul = 1;
    // in the case of interleaved RGB output, write only to single channel, but
    // 3 samples at once
    if (params.fmt == NVJPEG_OUTPUT_RGBI || params.fmt == NVJPEG_OUTPUT_BGRI) {
      channels = 1;
      mul = 3;
    }
    // in the case of rgb create 3 buffers with sizes of original image
    else if (params.fmt == NVJPEG_OUTPUT_RGB ||
             params.fmt == NVJPEG_OUTPUT_BGR) {
      channels = 3;
      widths[1] = widths[2] = widths[0];
      heights[1] = heights[2] = heights[0];
    }

    // realloc output buffer if required
    for (int c = 0; c < channels; c++) {
      int aw = mul * widths[c];
      int ah = heights[c];
      int sz = aw * ah;
      ibuf[i].pitch[c] = aw;
      if (sz > isz[i].pitch[c]) {
        if (ibuf[i].channel[c]) {
          checkCudaErrors(cudaFree(ibuf[i].channel[c]));
        }
        checkCudaErrors(cudaMalloc(&ibuf[i].channel[c], sz));
        isz[i].pitch[c] = sz;
      }
    }
  }
  return EXIT_SUCCESS;
}

void release_buffers(std::vector<nvjpegImage_t> &ibuf) {
  for (int i = 0; i < ibuf.size(); i++) {
    for (int c = 0; c < NVJPEG_MAX_COMPONENT; c++)
      if (ibuf[i].channel[c]) checkCudaErrors(cudaFree(ibuf[i].channel[c]));
  }
}

int decode_images(const FileData &img_data, const std::vector<size_t> &img_len,
                  std::vector<nvjpegImage_t> &out, decode_params_t &params,
                  double &time) {
  checkCudaErrors(cudaStreamSynchronize(params.stream));
  nvjpegStatus_t err;
  StopWatchInterface *timer = NULL;
  sdkCreateTimer(&timer);

  if (!params.batched) {
    if (!params.pipelined)  // decode one image at a time
    {
      int thread_idx = 0;
      sdkStartTimer(&timer);
      for (int i = 0; i < params.batch_size; i++) {
        checkCudaErrors(nvjpegDecode(params.nvjpeg_handle, params.nvjpeg_state,
                                     (const unsigned char *)img_data[i].data(),
                                     img_len[i], params.fmt, &out[i],
                                     params.stream));
        checkCudaErrors(cudaStreamSynchronize(params.stream));
      }
    } else {
      int thread_idx = 0;
      sdkStartTimer(&timer);
      for (int i = 0; i < params.batch_size; i++) {
        checkCudaErrors(
            nvjpegDecodePhaseOne(params.nvjpeg_handle, params.nvjpeg_state,
                                 (const unsigned char *)img_data[i].data(),
                                 img_len[i], params.fmt, params.stream));
        checkCudaErrors(cudaStreamSynchronize(params.stream));
        checkCudaErrors(nvjpegDecodePhaseTwo(
            params.nvjpeg_handle, params.nvjpeg_state, params.stream));
        checkCudaErrors(nvjpegDecodePhaseThree(
            params.nvjpeg_handle, params.nvjpeg_state, &out[i], params.stream));
      }
      checkCudaErrors(cudaStreamSynchronize(params.stream));
    }
  } else {
    std::vector<const unsigned char *> raw_inputs;
    for (int i = 0; i < params.batch_size; i++) {
      raw_inputs.push_back((const unsigned char *)img_data[i].data());
    }

    if (!params.pipelined)  // decode multiple images in a single batch
    {
      sdkStartTimer(&timer);
      checkCudaErrors(nvjpegDecodeBatched(
          params.nvjpeg_handle, params.nvjpeg_state, raw_inputs.data(),
          img_len.data(), out.data(), params.stream));
      checkCudaErrors(cudaStreamSynchronize(params.stream));
    } else {
      int thread_idx = 0;
      for (int i = 0; i < params.batch_size; i++) {
        checkCudaErrors(nvjpegDecodeBatchedPhaseOne(
            params.nvjpeg_handle, params.nvjpeg_state, raw_inputs[i],
            img_len[i], i, thread_idx, params.stream));
      }
      checkCudaErrors(nvjpegDecodeBatchedPhaseTwo(
          params.nvjpeg_handle, params.nvjpeg_state, params.stream));
      checkCudaErrors(nvjpegDecodeBatchedPhaseThree(params.nvjpeg_handle,
                                                    params.nvjpeg_state,
                                                    out.data(), params.stream));
      checkCudaErrors(cudaStreamSynchronize(params.stream));
    }
  }
  sdkStopTimer(&timer);
  time = sdkGetAverageTimerValue(&timer)/1000.0f;

  return EXIT_SUCCESS;
}

int write_images(std::vector<nvjpegImage_t> &iout, std::vector<int> &widths,
                 std::vector<int> &heights, decode_params_t &params,
                 FileNames &filenames) {
  for (int i = 0; i < params.batch_size; i++) {
    // Get the file name, without extension.
    // This will be used to rename the output file.
    size_t position = filenames[i].rfind("/");
    std::string sFileName =
        (std::string::npos == position)
            ? filenames[i]
            : filenames[i].substr(position + 1, filenames[i].size());
    position = sFileName.rfind(".");
    sFileName = (std::string::npos == position) ? sFileName
                                                : sFileName.substr(0, position);
    std::string fname(params.output_dir + "./" + sFileName + ".bmp");

    int err;
    if (params.fmt == NVJPEG_OUTPUT_RGB || params.fmt == NVJPEG_OUTPUT_BGR) {
      err = writeBMP(fname.c_str(), iout[i].channel[0], iout[i].pitch[0],
                     iout[i].channel[1], iout[i].pitch[1], iout[i].channel[2],
                     iout[i].pitch[2], widths[i], heights[i]);
    } else if (params.fmt == NVJPEG_OUTPUT_RGBI ||
               params.fmt == NVJPEG_OUTPUT_BGRI) {
      // Write BMP from interleaved data
      err = writeBMPi(fname.c_str(), iout[i].channel[0], iout[i].pitch[0],
                      widths[i], heights[i]);
    }
    if (err) {
      std::cout << "Cannot write output file: " << fname << std::endl;
      return EXIT_FAILURE;
    }
    std::cout << "Done writing decoded image to file: " << fname << std::endl;
  }
}

double process_images(FileNames &image_names, decode_params_t &params,
                      double &total) {
  // vector for storing raw files and file lengths
  FileData file_data(params.batch_size);
  std::vector<size_t> file_len(params.batch_size);
  FileNames current_names(params.batch_size);
  std::vector<int> widths(params.batch_size);
  std::vector<int> heights(params.batch_size);
  // we wrap over image files to process total_images of files
  FileNames::iterator file_iter = image_names.begin();

  // stream for decoding
  checkCudaErrors(
      cudaStreamCreateWithFlags(&params.stream, cudaStreamNonBlocking));

  int total_processed = 0;

  // output buffers
  std::vector<nvjpegImage_t> iout(params.batch_size);
  // output buffer sizes, for convenience
  std::vector<nvjpegImage_t> isz(params.batch_size);

  for (int i = 0; i < iout.size(); i++) {
    for (int c = 0; c < NVJPEG_MAX_COMPONENT; c++) {
      iout[i].channel[c] = NULL;
      iout[i].pitch[c] = 0;
      isz[i].pitch[c] = 0;
    }
  }

  double test_time = 0;
  int warmup = 0;
  while (total_processed < params.total_images) {
    if (read_next_batch(image_names, params.batch_size, file_iter, file_data,
                        file_len, current_names))
      return EXIT_FAILURE;

    if (prepare_buffers(file_data, file_len, widths, heights, iout, isz,
                        current_names, params))
      return EXIT_FAILURE;

    double time;
    if (decode_images(file_data, file_len, iout, params, time))
      return EXIT_FAILURE;

      uchar *matptr = new uchar[widths[0]*heights[0]*3];
      cv::Mat img(heights[0], widths[0], CV_8UC3, matptr);
      uchar* ptr = matptr;
      for(int i=0;i<3;i++) {
          int width = widths[0];
          int height = heights[0];
          checkCudaErrors(cudaMemcpy2D(ptr, width, iout[0].channel[i], iout[0].pitch[i], width, height, cudaMemcpyDeviceToHost));
          checkCudaErrors(cudaStreamSynchronize(params.stream));
          ptr += width * height;
      }
      cv::imshow("", img);
      cv::waitKey(0);
    if (warmup < params.warmup) {
      warmup++;
    } else {
      total_processed += params.batch_size;
      test_time += time;
    }

    if (params.write_decoded)
      write_images(iout, widths, heights, params, current_names);
  }
  total = test_time;

  release_buffers(iout);

  checkCudaErrors(cudaStreamDestroy(params.stream));

  return EXIT_SUCCESS;
}

// parse parameters
int findParamIndex(const char **argv, int argc, const char *parm) {
  int count = 0;
  int index = -1;

  for (int i = 0; i < argc; i++) {
    if (strncmp(argv[i], parm, 100) == 0) {
      index = i;
      count++;
    }
  }

  if (count == 0 || count == 1) {
    return index;
  } else {
    std::cout << "Error, parameter " << parm
              << " has been specified more than once, exiting\n"
              << std::endl;
    return -1;
  }

  return -1;
}

int main(int argc, const char *argv[]) {
  decode_params_t params;

  params.input_dir = "/home/taotao/Documents/nvjpeg/imgs/";
  params.batch_size = 1;
  params.total_images = 1;
  params.dev = 0;
  params.dev = findCudaDevice(argc, argv);
  params.warmup = 0;
  params.batched = false;
  params.pipelined = false;
  params.fmt = NVJPEG_OUTPUT_RGB;
  params.write_decoded = true;
//  cudaDeviceProp props;
//  checkCudaErrors(cudaGetDeviceProperties(&props, params.dev));
//
//  printf("Using GPU %d (%s, %d SMs, %d th/SM max, CC %d.%d, ECC %s)\n",
//         params.dev, props.name, props.multiProcessorCount,
//         props.maxThreadsPerMultiProcessor, props.major, props.minor,
//         props.ECCEnabled ? "on" : "off");

//  nvjpegDevAllocator_t dev_allocator = {&dev_malloc, &dev_free};
//  checkCudaErrors(nvjpegCreate(NVJPEG_BACKEND_DEFAULT, &dev_allocator,
//                               &params.nvjpeg_handle));
  checkCudaErrors(nvjpegCreateSimple(&params.nvjpeg_handle));
  checkCudaErrors(
      nvjpegJpegStateCreate(params.nvjpeg_handle, &params.nvjpeg_state));
  checkCudaErrors(
      nvjpegDecodeBatchedInitialize(params.nvjpeg_handle, params.nvjpeg_state,
                                    params.batch_size, 1, params.fmt));

  // read source images
  FileNames image_names;
  readInput(params.input_dir, image_names);

  if (params.total_images == -1) {
    params.total_images = image_names.size();
  } else if (params.total_images % params.batch_size) {
    params.total_images =
        ((params.total_images) / params.batch_size) * params.batch_size;
    std::cout << "Changing total_images number to " << params.total_images
              << " to be multiple of batch_size - " << params.batch_size
              << std::endl;
  }

  std::cout << "Decoding images in directory: " << params.input_dir
            << ", total " << params.total_images << ", batchsize "
            << params.batch_size << std::endl;

  double total;
  if (process_images(image_names, params, total)) return EXIT_FAILURE;
  std::cout << "Total decoding time: " << total << std::endl;
  std::cout << "Avg decoding time per image: " << total / params.total_images
            << std::endl;
  std::cout << "Avg images per sec: " << params.total_images / total
            << std::endl;
  std::cout << "Avg decoding time per batch: "
            << total / ((params.total_images + params.batch_size - 1) /
                        params.batch_size)
            << std::endl;

  checkCudaErrors(nvjpegJpegStateDestroy(params.nvjpeg_state));
  checkCudaErrors(nvjpegDestroy(params.nvjpeg_handle));

  return EXIT_SUCCESS;
}







/* EOF */


cmake_minimum_required(VERSION 3.13)

project(mpi-cuda-waves)

find_package(MPI)

enable_language(CUDA)
add_compile_options()
add_link_options()

file(GLOB SRC "src/*.cu")
add_executable(${PROJECT_NAME} ${SRC})
target_link_libraries(${PROJECT_NAME} MPI::MPI_C m)

install(TARGETS ${PROJECT_NAME} DESTINATION /usr/bin)

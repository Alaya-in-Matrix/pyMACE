cmake_minimum_required(VERSION 3.2.1)
project(cec2014expensive)

set(SRC cec14_eotp.cpp CEC14_EOTP_C.cpp)
set(EXE cec2014expensive)
add_executable(${EXE} ${SRC})

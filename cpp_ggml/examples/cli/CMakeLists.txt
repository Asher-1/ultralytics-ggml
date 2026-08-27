# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# yolo-cli: single-image inference commands for every task.
add_executable(yolo-cli ${CMAKE_CURRENT_SOURCE_DIR}/../../src/cli.cpp)
target_link_libraries(yolo-cli PRIVATE yologgml)
target_compile_definitions(yolo-cli PRIVATE YOLO_GGML_CLIP=$<BOOL:${YOLO_GGML_CLIP}>)

# End-to-end solution scenarios (each pairs one task family with an official
# Ultralytics solutions scenario; see the per-file header comments).
add_executable(yolo-region-count yolo_region_count.cpp examples_common.cpp)
target_link_libraries(yolo-region-count PRIVATE yologgml)

add_executable(yolo-crop yolo_crop.cpp examples_common.cpp)
target_link_libraries(yolo-crop PRIVATE yologgml)

add_executable(yolo-workout yolo_workout.cpp examples_common.cpp)
target_link_libraries(yolo-workout PRIVATE yologgml)

add_executable(yolo-obb yolo_obb.cpp examples_common.cpp)
target_link_libraries(yolo-obb PRIVATE yologgml)

add_executable(yolo-city yolo_city.cpp examples_common.cpp)
target_link_libraries(yolo-city PRIVATE yologgml)

add_executable(yolo-topk yolo_topk.cpp examples_common.cpp)
target_link_libraries(yolo-topk PRIVATE yologgml)

add_executable(yolo-distance yolo_distance.cpp examples_common.cpp)
target_link_libraries(yolo-distance PRIVATE yologgml)

# CLIP-based semantic image search (requires YOLO_GGML_CLIP).
if(YOLO_GGML_CLIP)
    add_executable(yolo-similarity yolo_similarity.cpp)
    target_link_libraries(yolo-similarity PRIVATE yologgml)
endif()

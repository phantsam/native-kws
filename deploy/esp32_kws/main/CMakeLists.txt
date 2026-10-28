idf_build_get_property(component_targets __COMPONENT_TARGETS)
if ("___idf_espressif__esp-dl" IN_LIST component_targets)
   idf_component_get_property(espdl_dir espressif__esp-dl COMPONENT_DIR)
elseif("___idf_esp-dl" IN_LIST component_targets)
   idf_component_get_property(espdl_dir esp-dl COMPONENT_DIR)
endif()
set(cmake_dir ${espdl_dir}/fbs_loader/cmake)
include(${cmake_dir}/utilities.cmake)

idf_component_register(SRCS app_main.cpp
                       INCLUDE_DIRS . ../../onchip/bcresnet-2
                       REQUIRES esp-dl esp_timer)

# Model stays in flash .rodata (symbol _binary_model_espdl_start).
target_add_aligned_binary_data(${COMPONENT_LIB} ${CMAKE_CURRENT_SOURCE_DIR}/models/model.espdl BINARY)

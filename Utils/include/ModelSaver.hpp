// use boost to save model to file

#ifndef UTILS_MODEL_SAVER_HPP
#define UTILS_MODEL_SAVER_HPP

#include <string>
#include <vector>
#include <fstream>
#include <boost/serialization/vector.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/export.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/assume_abstract.hpp>
#include <boost/serialization/split_member.hpp>

namespace Utils {
    class ModelSaver {
    public:
        ModelSaver() = delete;
        ModelSaver(const ModelSaver&) = delete;
        ModelSaver& operator=(const ModelSaver&) = delete;
        ModelSaver(ModelSaver&&) = delete;
        ModelSaver& operator=(ModelSaver&&) = delete;

        template <typename T>
        static void saveModel(const T& model, const std::string& filename) {
            std::ofstream ofs(filename);
            boost::archive::text_oarchive oa(ofs);
            oa << model;
        }

        template <typename T>
        static void loadModel(T& model, const std::string& filename) {
            std::ifstream ifs(filename);
            boost::archive::text_iarchive ia(ifs);
            ia >> model;
        }

        template <typename T>
        static T loadModel(const std::string& filename) {
            T model;
            std::ifstream ifs(filename);
            boost::archive::text_iarchive ia(ifs);
            ia >> model;
            return model;
        }
        
    };
}

#endif
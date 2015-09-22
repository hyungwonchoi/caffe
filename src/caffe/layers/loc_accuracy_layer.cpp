#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void LocAccuracyLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  //top_k_ = this->layer_param_.accuracy_param().top_k();

  //has_ignore_label_ =
   // this->layer_param_.accuracy_param().has_ignore_label();
  //if (has_ignore_label_) {
   // ignore_label_ = this->layer_param_.accuracy_param().ignore_label();
  //}
}

template <typename Dtype>
void LocAccuracyLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  //CHECK_LE(top_k_, bottom[0]->count() / bottom[1]->count())
  //    << "top_k must be less than or equal to the number of classes.";
  label_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.accuracy_param().axis());
  outer_num_ = bottom[0]->count(0, label_axis_);
  inner_num_ = bottom[0]->count(label_axis_ );
  //std::cout<< bottom[0]->shape(0) << std::endl;
  //std::cout<< ":" << bottom[0]->shape(0)<<":"<<bottom[0]->shape(1)<<":"
  //<<bottom[0]->count()<<":"<< label_axis_<< std::endl;
  //std::cout << outer_num_ << "!!" << inner_num_ << std::endl;
  CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if label axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";
  vector<int> top_shape(0);  // Accuracy is a scalar; 0 axes.
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void LocAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype accuracy = 0;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  const int dim = bottom[0]->count() / outer_num_;
  //const int num_labels = bottom[0]->shape(label_axis_);
  //std::cout<< dim << ":" << num_labels << ":" << bottom[0]->count()
  //<<":"<< outer_num_<<":"<<inner_num_<<std::endl;
  //vector<Dtype> maxval(top_k_+1);
  //vector<int> max_id(top_k_+1);
  int count = 0;
  for (int i = 0; i < outer_num_; ++i) {
  //for (int i = 0; i < 1; ++i) {
    //for (int j = 0; j < inner_num_; ++j) {
      
      float x11 = bottom_data[i * dim + 0];
      float x12 = bottom_data[i * dim + 2];
      float y11 = bottom_data[i * dim + 1];
      float y12 = bottom_data[i * dim + 3];
      float x21 = bottom_label[i * dim + 0];
      float x22 = bottom_label[i * dim + 2];
      float y21 = bottom_label[i * dim + 1];
      float y22 = bottom_label[i * dim + 3];
      //std::cout << "(" << x11 << "," << y11 << "," << x12 << "," << y12 << std::endl;
      //std::cout << "(" << x21 << "," << y21 << "," << x22 << "," << y22 << std::endl;

      float x_overlap = std::max(0.0f, std::min(x12, x22) - std::max(x11, x21));
      float y_overlap = std::max(0.0f, std::min(y12, y22) - std::max(y11, y21));
      float overlapArea = x_overlap * y_overlap;
      float unionArea = (x12-x11)*(y12-y11)+(x22-x21)*(y22-y21) - overlapArea;
      //std::cout << x_overlap << "," << y_overlap << std::endl;
      //std::cout << overlapArea << ":" << unionArea << std::endl;
      if (overlapArea/unionArea >= 0.5)
        ++accuracy;

      ++count;
    //}
  }

  // LOG(INFO) << "Accuracy: " << accuracy;
  top[0]->mutable_cpu_data()[0] = accuracy / count;
  // Accuracy layer should not be used as a loss function.
}

INSTANTIATE_CLASS(LocAccuracyLayer);
REGISTER_LAYER_CLASS(LocAccuracy);

}  // namespace caffe

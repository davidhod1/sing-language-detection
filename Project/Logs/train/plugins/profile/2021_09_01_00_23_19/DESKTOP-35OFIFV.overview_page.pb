?	?Fv?e$'@?Fv?e$'@!?Fv?e$'@      ??!       "q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?Fv?e$'@?HV???1??z??b	@A?;FzQ??IJ?E??@r0*?G?z~X@)      =2T
Iterator::Root::ParallelMapV2??k????!),.???@)??k????1),.???@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat??ݯ|??!????h7@)? ??U??1??_?>E4@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate?}:3P??!]{_r;9@)?4?8EG??1????H82@:Preprocessing2E
Iterator::Root?QcB?%??!????uE@)?;?2Tń?1[	??L?$@:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSliceO#-??#|?!?1ê?@)O#-??#|?1?1ê?@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip?qS??! i54??L@)???ฌ{?1???"v@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?(??0i?!??l	@)?(??0i?1??l	@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap??.?u???!\p??ɐ;@)????b?1 K???@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"?69.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI<???$R@Q?#8?l;@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?HV????HV???!?HV???      ??!       "	??z??b	@??z??b	@!??z??b	@*      ??!       2	?;FzQ???;FzQ??!?;FzQ??:	J?E??@J?E??@!J?E??@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q<???$R@y?#8?l;@?"P
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop?)?~?o??!?)?~?o??0"&
CudnnRNNCudnnRNN=?1`#	??!n?]??".
IteratorGetNext/_10_Recv???,???!|jO????"m
:categorical_crossentropy/softmax_cross_entropy_with_logitsSoftmaxCrossEntropyWithLogitsR??֓?!]%L9??"*
transpose_0	Transpose3??U???!?aڼn???"9
 sequential/dropout/dropout/Mul_1MulGa],#???!{׋Io??"(

concat_1_0ConcatV2a픔*??!1+??so??";
gradients/split_2_grad/concatConcatV2xL0X?ځ?!c?>?߶??"C
$gradients/transpose_9_grad/transpose	TransposeQo?k~?!B?N?????"*
transpose_9	TransposeQo?k~?!!_?0??Q      Y@Y???b:?@au?YL?W@q??H}AX@y?n^???"?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?69.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?97.0% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Pascal)(: B 
	?Fv?e$'@?Fv?e$'@!?Fv?e$'@      ??!       "q
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
	?HV????HV???!?HV???      ??!       "	??z??b	@??z??b	@!??z??b	@*      ??!       2	?;FzQ???;FzQ??!?;FzQ??:	J?E??@J?E??@!J?E??@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q<???$R@y?#8?l;@
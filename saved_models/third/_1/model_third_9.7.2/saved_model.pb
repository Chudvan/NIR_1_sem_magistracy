??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.7.02v2.7.0-rc1-69-gc256c071bb28??
~
dense_1024/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	*"
shared_namedense_1024/kernel
w
%dense_1024/kernel/Read/ReadVariableOpReadVariableOpdense_1024/kernel*
_output_shapes

:	*
dtype0
v
dense_1024/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	* 
shared_namedense_1024/bias
o
#dense_1024/bias/Read/ReadVariableOpReadVariableOpdense_1024/bias*
_output_shapes
:	*
dtype0
~
dense_1025/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	*"
shared_namedense_1025/kernel
w
%dense_1025/kernel/Read/ReadVariableOpReadVariableOpdense_1025/kernel*
_output_shapes

:	*
dtype0
v
dense_1025/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namedense_1025/bias
o
#dense_1025/bias/Read/ReadVariableOpReadVariableOpdense_1025/bias*
_output_shapes
:*
dtype0
~
dense_1026/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*"
shared_namedense_1026/kernel
w
%dense_1026/kernel/Read/ReadVariableOpReadVariableOpdense_1026/kernel*
_output_shapes

:*
dtype0
v
dense_1026/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namedense_1026/bias
o
#dense_1026/bias/Read/ReadVariableOpReadVariableOpdense_1026/bias*
_output_shapes
:*
dtype0
~
dense_1027/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*"
shared_namedense_1027/kernel
w
%dense_1027/kernel/Read/ReadVariableOpReadVariableOpdense_1027/kernel*
_output_shapes

:*
dtype0
v
dense_1027/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namedense_1027/bias
o
#dense_1027/bias/Read/ReadVariableOpReadVariableOpdense_1027/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
?
Adam/dense_1024/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	*)
shared_nameAdam/dense_1024/kernel/m
?
,Adam/dense_1024/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1024/kernel/m*
_output_shapes

:	*
dtype0
?
Adam/dense_1024/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*'
shared_nameAdam/dense_1024/bias/m
}
*Adam/dense_1024/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1024/bias/m*
_output_shapes
:	*
dtype0
?
Adam/dense_1025/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	*)
shared_nameAdam/dense_1025/kernel/m
?
,Adam/dense_1025/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1025/kernel/m*
_output_shapes

:	*
dtype0
?
Adam/dense_1025/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/dense_1025/bias/m
}
*Adam/dense_1025/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1025/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_1026/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*)
shared_nameAdam/dense_1026/kernel/m
?
,Adam/dense_1026/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1026/kernel/m*
_output_shapes

:*
dtype0
?
Adam/dense_1026/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/dense_1026/bias/m
}
*Adam/dense_1026/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1026/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_1027/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*)
shared_nameAdam/dense_1027/kernel/m
?
,Adam/dense_1027/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1027/kernel/m*
_output_shapes

:*
dtype0
?
Adam/dense_1027/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/dense_1027/bias/m
}
*Adam/dense_1027/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1027/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_1024/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	*)
shared_nameAdam/dense_1024/kernel/v
?
,Adam/dense_1024/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1024/kernel/v*
_output_shapes

:	*
dtype0
?
Adam/dense_1024/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*'
shared_nameAdam/dense_1024/bias/v
}
*Adam/dense_1024/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1024/bias/v*
_output_shapes
:	*
dtype0
?
Adam/dense_1025/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	*)
shared_nameAdam/dense_1025/kernel/v
?
,Adam/dense_1025/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1025/kernel/v*
_output_shapes

:	*
dtype0
?
Adam/dense_1025/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/dense_1025/bias/v
}
*Adam/dense_1025/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1025/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense_1026/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*)
shared_nameAdam/dense_1026/kernel/v
?
,Adam/dense_1026/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1026/kernel/v*
_output_shapes

:*
dtype0
?
Adam/dense_1026/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/dense_1026/bias/v
}
*Adam/dense_1026/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1026/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense_1027/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*)
shared_nameAdam/dense_1027/kernel/v
?
,Adam/dense_1027/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1027/kernel/v*
_output_shapes

:*
dtype0
?
Adam/dense_1027/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/dense_1027/bias/v
}
*Adam/dense_1027/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1027/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?,
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?+
value?+B?+ B?+
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
	optimizer
	variables
trainable_variables
	regularization_losses

	keras_api

signatures
 
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
 	variables
!trainable_variables
"regularization_losses
#	keras_api
?
$iter

%beta_1

&beta_2
	'decay
(learning_ratemGmHmImJmKmLmMmNvOvPvQvRvSvTvUvV
8
0
1
2
3
4
5
6
7
8
0
1
2
3
4
5
6
7
 
?
)non_trainable_variables

*layers
+metrics
,layer_regularization_losses
-layer_metrics
	variables
trainable_variables
	regularization_losses
 
][
VARIABLE_VALUEdense_1024/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEdense_1024/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
.non_trainable_variables

/layers
0metrics
1layer_regularization_losses
2layer_metrics
	variables
trainable_variables
regularization_losses
][
VARIABLE_VALUEdense_1025/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEdense_1025/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
3non_trainable_variables

4layers
5metrics
6layer_regularization_losses
7layer_metrics
	variables
trainable_variables
regularization_losses
][
VARIABLE_VALUEdense_1026/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEdense_1026/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
8non_trainable_variables

9layers
:metrics
;layer_regularization_losses
<layer_metrics
	variables
trainable_variables
regularization_losses
][
VARIABLE_VALUEdense_1027/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEdense_1027/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
=non_trainable_variables

>layers
?metrics
@layer_regularization_losses
Alayer_metrics
 	variables
!trainable_variables
"regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
#
0
1
2
3
4

B0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	Ctotal
	Dcount
E	variables
F	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

C0
D1

E	variables
?~
VARIABLE_VALUEAdam/dense_1024/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dense_1024/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/dense_1025/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dense_1025/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/dense_1026/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dense_1026/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/dense_1027/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dense_1027/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/dense_1024/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dense_1024/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/dense_1025/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dense_1025/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/dense_1026/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dense_1026/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/dense_1027/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dense_1027/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|
serving_default_input_257Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_257dense_1024/kerneldense_1024/biasdense_1025/kerneldense_1025/biasdense_1026/kerneldense_1026/biasdense_1027/kerneldense_1027/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? */
f*R(
&__inference_signature_wrapper_18664567
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename%dense_1024/kernel/Read/ReadVariableOp#dense_1024/bias/Read/ReadVariableOp%dense_1025/kernel/Read/ReadVariableOp#dense_1025/bias/Read/ReadVariableOp%dense_1026/kernel/Read/ReadVariableOp#dense_1026/bias/Read/ReadVariableOp%dense_1027/kernel/Read/ReadVariableOp#dense_1027/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp,Adam/dense_1024/kernel/m/Read/ReadVariableOp*Adam/dense_1024/bias/m/Read/ReadVariableOp,Adam/dense_1025/kernel/m/Read/ReadVariableOp*Adam/dense_1025/bias/m/Read/ReadVariableOp,Adam/dense_1026/kernel/m/Read/ReadVariableOp*Adam/dense_1026/bias/m/Read/ReadVariableOp,Adam/dense_1027/kernel/m/Read/ReadVariableOp*Adam/dense_1027/bias/m/Read/ReadVariableOp,Adam/dense_1024/kernel/v/Read/ReadVariableOp*Adam/dense_1024/bias/v/Read/ReadVariableOp,Adam/dense_1025/kernel/v/Read/ReadVariableOp*Adam/dense_1025/bias/v/Read/ReadVariableOp,Adam/dense_1026/kernel/v/Read/ReadVariableOp*Adam/dense_1026/bias/v/Read/ReadVariableOp,Adam/dense_1027/kernel/v/Read/ReadVariableOp*Adam/dense_1027/bias/v/Read/ReadVariableOpConst*,
Tin%
#2!	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__traced_save_18664869
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_1024/kerneldense_1024/biasdense_1025/kerneldense_1025/biasdense_1026/kerneldense_1026/biasdense_1027/kerneldense_1027/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/dense_1024/kernel/mAdam/dense_1024/bias/mAdam/dense_1025/kernel/mAdam/dense_1025/bias/mAdam/dense_1026/kernel/mAdam/dense_1026/bias/mAdam/dense_1027/kernel/mAdam/dense_1027/bias/mAdam/dense_1024/kernel/vAdam/dense_1024/bias/vAdam/dense_1025/kernel/vAdam/dense_1025/bias/vAdam/dense_1026/kernel/vAdam/dense_1026/bias/vAdam/dense_1027/kernel/vAdam/dense_1027/bias/v*+
Tin$
"2 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference__traced_restore_18664972??
?

?
H__inference_dense_1027_layer_call_and_return_conditional_losses_18664753

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
-__inference_dense_1024_layer_call_fn_18664682

inputs
unknown:	
	unknown_0:	
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_dense_1024_layer_call_and_return_conditional_losses_18664286o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
H__inference_dense_1024_layer_call_and_return_conditional_losses_18664693

inputs0
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:	
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????	a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????	w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?}
?
$__inference__traced_restore_18664972
file_prefix4
"assignvariableop_dense_1024_kernel:	0
"assignvariableop_1_dense_1024_bias:	6
$assignvariableop_2_dense_1025_kernel:	0
"assignvariableop_3_dense_1025_bias:6
$assignvariableop_4_dense_1026_kernel:0
"assignvariableop_5_dense_1026_bias:6
$assignvariableop_6_dense_1027_kernel:0
"assignvariableop_7_dense_1027_bias:&
assignvariableop_8_adam_iter:	 (
assignvariableop_9_adam_beta_1: )
assignvariableop_10_adam_beta_2: (
assignvariableop_11_adam_decay: 0
&assignvariableop_12_adam_learning_rate: #
assignvariableop_13_total: #
assignvariableop_14_count: >
,assignvariableop_15_adam_dense_1024_kernel_m:	8
*assignvariableop_16_adam_dense_1024_bias_m:	>
,assignvariableop_17_adam_dense_1025_kernel_m:	8
*assignvariableop_18_adam_dense_1025_bias_m:>
,assignvariableop_19_adam_dense_1026_kernel_m:8
*assignvariableop_20_adam_dense_1026_bias_m:>
,assignvariableop_21_adam_dense_1027_kernel_m:8
*assignvariableop_22_adam_dense_1027_bias_m:>
,assignvariableop_23_adam_dense_1024_kernel_v:	8
*assignvariableop_24_adam_dense_1024_bias_v:	>
,assignvariableop_25_adam_dense_1025_kernel_v:	8
*assignvariableop_26_adam_dense_1025_bias_v:>
,assignvariableop_27_adam_dense_1026_kernel_v:8
*assignvariableop_28_adam_dense_1026_bias_v:>
,assignvariableop_29_adam_dense_1027_kernel_v:8
*assignvariableop_30_adam_dense_1027_bias_v:
identity_32??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::*.
dtypes$
"2 	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp"assignvariableop_dense_1024_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp"assignvariableop_1_dense_1024_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp$assignvariableop_2_dense_1025_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp"assignvariableop_3_dense_1025_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp$assignvariableop_4_dense_1026_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp"assignvariableop_5_dense_1026_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp$assignvariableop_6_dense_1027_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp"assignvariableop_7_dense_1027_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_beta_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp&assignvariableop_12_adam_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp,assignvariableop_15_adam_dense_1024_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp*assignvariableop_16_adam_dense_1024_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp,assignvariableop_17_adam_dense_1025_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp*assignvariableop_18_adam_dense_1025_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp,assignvariableop_19_adam_dense_1026_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp*assignvariableop_20_adam_dense_1026_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp,assignvariableop_21_adam_dense_1027_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp*assignvariableop_22_adam_dense_1027_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp,assignvariableop_23_adam_dense_1024_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp*assignvariableop_24_adam_dense_1024_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp,assignvariableop_25_adam_dense_1025_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOp*assignvariableop_26_adam_dense_1025_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOp,assignvariableop_27_adam_dense_1026_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOp*assignvariableop_28_adam_dense_1026_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOp,assignvariableop_29_adam_dense_1027_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOp*assignvariableop_30_adam_dense_1027_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_31Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_32IdentityIdentity_31:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_32Identity_32:output:0*S
_input_shapesB
@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?	
?
,__inference_model_256_layer_call_fn_18664490
	input_257
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall	input_257unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_model_256_layer_call_and_return_conditional_losses_18664450o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
'
_output_shapes
:?????????
#
_user_specified_name	input_257
?	
?
,__inference_model_256_layer_call_fn_18664363
	input_257
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall	input_257unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_model_256_layer_call_and_return_conditional_losses_18664344o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
'
_output_shapes
:?????????
#
_user_specified_name	input_257
?	
?
,__inference_model_256_layer_call_fn_18664588

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_model_256_layer_call_and_return_conditional_losses_18664344o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?%
?
G__inference_model_256_layer_call_and_return_conditional_losses_18664641

inputs;
)dense_1024_matmul_readvariableop_resource:	8
*dense_1024_biasadd_readvariableop_resource:	;
)dense_1025_matmul_readvariableop_resource:	8
*dense_1025_biasadd_readvariableop_resource:;
)dense_1026_matmul_readvariableop_resource:8
*dense_1026_biasadd_readvariableop_resource:;
)dense_1027_matmul_readvariableop_resource:8
*dense_1027_biasadd_readvariableop_resource:
identity??!dense_1024/BiasAdd/ReadVariableOp? dense_1024/MatMul/ReadVariableOp?!dense_1025/BiasAdd/ReadVariableOp? dense_1025/MatMul/ReadVariableOp?!dense_1026/BiasAdd/ReadVariableOp? dense_1026/MatMul/ReadVariableOp?!dense_1027/BiasAdd/ReadVariableOp? dense_1027/MatMul/ReadVariableOp?
 dense_1024/MatMul/ReadVariableOpReadVariableOp)dense_1024_matmul_readvariableop_resource*
_output_shapes

:	*
dtype0
dense_1024/MatMulMatMulinputs(dense_1024/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	?
!dense_1024/BiasAdd/ReadVariableOpReadVariableOp*dense_1024_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0?
dense_1024/BiasAddBiasAdddense_1024/MatMul:product:0)dense_1024/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	f
dense_1024/ReluReludense_1024/BiasAdd:output:0*
T0*'
_output_shapes
:?????????	?
 dense_1025/MatMul/ReadVariableOpReadVariableOp)dense_1025_matmul_readvariableop_resource*
_output_shapes

:	*
dtype0?
dense_1025/MatMulMatMuldense_1024/Relu:activations:0(dense_1025/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
!dense_1025/BiasAdd/ReadVariableOpReadVariableOp*dense_1025_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_1025/BiasAddBiasAdddense_1025/MatMul:product:0)dense_1025/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f
dense_1025/ReluReludense_1025/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
 dense_1026/MatMul/ReadVariableOpReadVariableOp)dense_1026_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
dense_1026/MatMulMatMuldense_1025/Relu:activations:0(dense_1026/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
!dense_1026/BiasAdd/ReadVariableOpReadVariableOp*dense_1026_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_1026/BiasAddBiasAdddense_1026/MatMul:product:0)dense_1026/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f
dense_1026/ReluReludense_1026/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
 dense_1027/MatMul/ReadVariableOpReadVariableOp)dense_1027_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
dense_1027/MatMulMatMuldense_1026/Relu:activations:0(dense_1027/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
!dense_1027/BiasAdd/ReadVariableOpReadVariableOp*dense_1027_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_1027/BiasAddBiasAdddense_1027/MatMul:product:0)dense_1027/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????l
dense_1027/SigmoidSigmoiddense_1027/BiasAdd:output:0*
T0*'
_output_shapes
:?????????e
IdentityIdentitydense_1027/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp"^dense_1024/BiasAdd/ReadVariableOp!^dense_1024/MatMul/ReadVariableOp"^dense_1025/BiasAdd/ReadVariableOp!^dense_1025/MatMul/ReadVariableOp"^dense_1026/BiasAdd/ReadVariableOp!^dense_1026/MatMul/ReadVariableOp"^dense_1027/BiasAdd/ReadVariableOp!^dense_1027/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 2F
!dense_1024/BiasAdd/ReadVariableOp!dense_1024/BiasAdd/ReadVariableOp2D
 dense_1024/MatMul/ReadVariableOp dense_1024/MatMul/ReadVariableOp2F
!dense_1025/BiasAdd/ReadVariableOp!dense_1025/BiasAdd/ReadVariableOp2D
 dense_1025/MatMul/ReadVariableOp dense_1025/MatMul/ReadVariableOp2F
!dense_1026/BiasAdd/ReadVariableOp!dense_1026/BiasAdd/ReadVariableOp2D
 dense_1026/MatMul/ReadVariableOp dense_1026/MatMul/ReadVariableOp2F
!dense_1027/BiasAdd/ReadVariableOp!dense_1027/BiasAdd/ReadVariableOp2D
 dense_1027/MatMul/ReadVariableOp dense_1027/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
,__inference_model_256_layer_call_fn_18664609

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_model_256_layer_call_and_return_conditional_losses_18664450o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
G__inference_model_256_layer_call_and_return_conditional_losses_18664450

inputs%
dense_1024_18664429:	!
dense_1024_18664431:	%
dense_1025_18664434:	!
dense_1025_18664436:%
dense_1026_18664439:!
dense_1026_18664441:%
dense_1027_18664444:!
dense_1027_18664446:
identity??"dense_1024/StatefulPartitionedCall?"dense_1025/StatefulPartitionedCall?"dense_1026/StatefulPartitionedCall?"dense_1027/StatefulPartitionedCall?
"dense_1024/StatefulPartitionedCallStatefulPartitionedCallinputsdense_1024_18664429dense_1024_18664431*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_dense_1024_layer_call_and_return_conditional_losses_18664286?
"dense_1025/StatefulPartitionedCallStatefulPartitionedCall+dense_1024/StatefulPartitionedCall:output:0dense_1025_18664434dense_1025_18664436*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_dense_1025_layer_call_and_return_conditional_losses_18664303?
"dense_1026/StatefulPartitionedCallStatefulPartitionedCall+dense_1025/StatefulPartitionedCall:output:0dense_1026_18664439dense_1026_18664441*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_dense_1026_layer_call_and_return_conditional_losses_18664320?
"dense_1027/StatefulPartitionedCallStatefulPartitionedCall+dense_1026/StatefulPartitionedCall:output:0dense_1027_18664444dense_1027_18664446*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_dense_1027_layer_call_and_return_conditional_losses_18664337z
IdentityIdentity+dense_1027/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp#^dense_1024/StatefulPartitionedCall#^dense_1025/StatefulPartitionedCall#^dense_1026/StatefulPartitionedCall#^dense_1027/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 2H
"dense_1024/StatefulPartitionedCall"dense_1024/StatefulPartitionedCall2H
"dense_1025/StatefulPartitionedCall"dense_1025/StatefulPartitionedCall2H
"dense_1026/StatefulPartitionedCall"dense_1026/StatefulPartitionedCall2H
"dense_1027/StatefulPartitionedCall"dense_1027/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
H__inference_dense_1024_layer_call_and_return_conditional_losses_18664286

inputs0
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:	
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????	a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????	w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
&__inference_signature_wrapper_18664567
	input_257
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall	input_257unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference__wrapped_model_18664268o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
'
_output_shapes
:?????????
#
_user_specified_name	input_257
?
?
-__inference_dense_1025_layer_call_fn_18664702

inputs
unknown:	
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_dense_1025_layer_call_and_return_conditional_losses_18664303o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????	: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs
?D
?
!__inference__traced_save_18664869
file_prefix0
,savev2_dense_1024_kernel_read_readvariableop.
*savev2_dense_1024_bias_read_readvariableop0
,savev2_dense_1025_kernel_read_readvariableop.
*savev2_dense_1025_bias_read_readvariableop0
,savev2_dense_1026_kernel_read_readvariableop.
*savev2_dense_1026_bias_read_readvariableop0
,savev2_dense_1027_kernel_read_readvariableop.
*savev2_dense_1027_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop7
3savev2_adam_dense_1024_kernel_m_read_readvariableop5
1savev2_adam_dense_1024_bias_m_read_readvariableop7
3savev2_adam_dense_1025_kernel_m_read_readvariableop5
1savev2_adam_dense_1025_bias_m_read_readvariableop7
3savev2_adam_dense_1026_kernel_m_read_readvariableop5
1savev2_adam_dense_1026_bias_m_read_readvariableop7
3savev2_adam_dense_1027_kernel_m_read_readvariableop5
1savev2_adam_dense_1027_bias_m_read_readvariableop7
3savev2_adam_dense_1024_kernel_v_read_readvariableop5
1savev2_adam_dense_1024_bias_v_read_readvariableop7
3savev2_adam_dense_1025_kernel_v_read_readvariableop5
1savev2_adam_dense_1025_bias_v_read_readvariableop7
3savev2_adam_dense_1026_kernel_v_read_readvariableop5
1savev2_adam_dense_1026_bias_v_read_readvariableop7
3savev2_adam_dense_1027_kernel_v_read_readvariableop5
1savev2_adam_dense_1027_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0,savev2_dense_1024_kernel_read_readvariableop*savev2_dense_1024_bias_read_readvariableop,savev2_dense_1025_kernel_read_readvariableop*savev2_dense_1025_bias_read_readvariableop,savev2_dense_1026_kernel_read_readvariableop*savev2_dense_1026_bias_read_readvariableop,savev2_dense_1027_kernel_read_readvariableop*savev2_dense_1027_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop3savev2_adam_dense_1024_kernel_m_read_readvariableop1savev2_adam_dense_1024_bias_m_read_readvariableop3savev2_adam_dense_1025_kernel_m_read_readvariableop1savev2_adam_dense_1025_bias_m_read_readvariableop3savev2_adam_dense_1026_kernel_m_read_readvariableop1savev2_adam_dense_1026_bias_m_read_readvariableop3savev2_adam_dense_1027_kernel_m_read_readvariableop1savev2_adam_dense_1027_bias_m_read_readvariableop3savev2_adam_dense_1024_kernel_v_read_readvariableop1savev2_adam_dense_1024_bias_v_read_readvariableop3savev2_adam_dense_1025_kernel_v_read_readvariableop1savev2_adam_dense_1025_bias_v_read_readvariableop3savev2_adam_dense_1026_kernel_v_read_readvariableop1savev2_adam_dense_1026_bias_v_read_readvariableop3savev2_adam_dense_1027_kernel_v_read_readvariableop1savev2_adam_dense_1027_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *.
dtypes$
"2 	?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes?
?: :	:	:	:::::: : : : : : : :	:	:	::::::	:	:	:::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:	: 

_output_shapes
:	:$ 

_output_shapes

:	: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:	: 

_output_shapes
:	:$ 

_output_shapes

:	: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:	: 

_output_shapes
:	:$ 

_output_shapes

:	: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
:: 

_output_shapes
: 
?
?
-__inference_dense_1026_layer_call_fn_18664722

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_dense_1026_layer_call_and_return_conditional_losses_18664320o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
H__inference_dense_1026_layer_call_and_return_conditional_losses_18664733

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?,
?
#__inference__wrapped_model_18664268
	input_257E
3model_256_dense_1024_matmul_readvariableop_resource:	B
4model_256_dense_1024_biasadd_readvariableop_resource:	E
3model_256_dense_1025_matmul_readvariableop_resource:	B
4model_256_dense_1025_biasadd_readvariableop_resource:E
3model_256_dense_1026_matmul_readvariableop_resource:B
4model_256_dense_1026_biasadd_readvariableop_resource:E
3model_256_dense_1027_matmul_readvariableop_resource:B
4model_256_dense_1027_biasadd_readvariableop_resource:
identity??+model_256/dense_1024/BiasAdd/ReadVariableOp?*model_256/dense_1024/MatMul/ReadVariableOp?+model_256/dense_1025/BiasAdd/ReadVariableOp?*model_256/dense_1025/MatMul/ReadVariableOp?+model_256/dense_1026/BiasAdd/ReadVariableOp?*model_256/dense_1026/MatMul/ReadVariableOp?+model_256/dense_1027/BiasAdd/ReadVariableOp?*model_256/dense_1027/MatMul/ReadVariableOp?
*model_256/dense_1024/MatMul/ReadVariableOpReadVariableOp3model_256_dense_1024_matmul_readvariableop_resource*
_output_shapes

:	*
dtype0?
model_256/dense_1024/MatMulMatMul	input_2572model_256/dense_1024/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	?
+model_256/dense_1024/BiasAdd/ReadVariableOpReadVariableOp4model_256_dense_1024_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0?
model_256/dense_1024/BiasAddBiasAdd%model_256/dense_1024/MatMul:product:03model_256/dense_1024/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	z
model_256/dense_1024/ReluRelu%model_256/dense_1024/BiasAdd:output:0*
T0*'
_output_shapes
:?????????	?
*model_256/dense_1025/MatMul/ReadVariableOpReadVariableOp3model_256_dense_1025_matmul_readvariableop_resource*
_output_shapes

:	*
dtype0?
model_256/dense_1025/MatMulMatMul'model_256/dense_1024/Relu:activations:02model_256/dense_1025/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
+model_256/dense_1025/BiasAdd/ReadVariableOpReadVariableOp4model_256_dense_1025_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model_256/dense_1025/BiasAddBiasAdd%model_256/dense_1025/MatMul:product:03model_256/dense_1025/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????z
model_256/dense_1025/ReluRelu%model_256/dense_1025/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
*model_256/dense_1026/MatMul/ReadVariableOpReadVariableOp3model_256_dense_1026_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
model_256/dense_1026/MatMulMatMul'model_256/dense_1025/Relu:activations:02model_256/dense_1026/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
+model_256/dense_1026/BiasAdd/ReadVariableOpReadVariableOp4model_256_dense_1026_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model_256/dense_1026/BiasAddBiasAdd%model_256/dense_1026/MatMul:product:03model_256/dense_1026/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????z
model_256/dense_1026/ReluRelu%model_256/dense_1026/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
*model_256/dense_1027/MatMul/ReadVariableOpReadVariableOp3model_256_dense_1027_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
model_256/dense_1027/MatMulMatMul'model_256/dense_1026/Relu:activations:02model_256/dense_1027/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
+model_256/dense_1027/BiasAdd/ReadVariableOpReadVariableOp4model_256_dense_1027_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model_256/dense_1027/BiasAddBiasAdd%model_256/dense_1027/MatMul:product:03model_256/dense_1027/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
model_256/dense_1027/SigmoidSigmoid%model_256/dense_1027/BiasAdd:output:0*
T0*'
_output_shapes
:?????????o
IdentityIdentity model_256/dense_1027/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp,^model_256/dense_1024/BiasAdd/ReadVariableOp+^model_256/dense_1024/MatMul/ReadVariableOp,^model_256/dense_1025/BiasAdd/ReadVariableOp+^model_256/dense_1025/MatMul/ReadVariableOp,^model_256/dense_1026/BiasAdd/ReadVariableOp+^model_256/dense_1026/MatMul/ReadVariableOp,^model_256/dense_1027/BiasAdd/ReadVariableOp+^model_256/dense_1027/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 2Z
+model_256/dense_1024/BiasAdd/ReadVariableOp+model_256/dense_1024/BiasAdd/ReadVariableOp2X
*model_256/dense_1024/MatMul/ReadVariableOp*model_256/dense_1024/MatMul/ReadVariableOp2Z
+model_256/dense_1025/BiasAdd/ReadVariableOp+model_256/dense_1025/BiasAdd/ReadVariableOp2X
*model_256/dense_1025/MatMul/ReadVariableOp*model_256/dense_1025/MatMul/ReadVariableOp2Z
+model_256/dense_1026/BiasAdd/ReadVariableOp+model_256/dense_1026/BiasAdd/ReadVariableOp2X
*model_256/dense_1026/MatMul/ReadVariableOp*model_256/dense_1026/MatMul/ReadVariableOp2Z
+model_256/dense_1027/BiasAdd/ReadVariableOp+model_256/dense_1027/BiasAdd/ReadVariableOp2X
*model_256/dense_1027/MatMul/ReadVariableOp*model_256/dense_1027/MatMul/ReadVariableOp:R N
'
_output_shapes
:?????????
#
_user_specified_name	input_257
?
?
G__inference_model_256_layer_call_and_return_conditional_losses_18664538
	input_257%
dense_1024_18664517:	!
dense_1024_18664519:	%
dense_1025_18664522:	!
dense_1025_18664524:%
dense_1026_18664527:!
dense_1026_18664529:%
dense_1027_18664532:!
dense_1027_18664534:
identity??"dense_1024/StatefulPartitionedCall?"dense_1025/StatefulPartitionedCall?"dense_1026/StatefulPartitionedCall?"dense_1027/StatefulPartitionedCall?
"dense_1024/StatefulPartitionedCallStatefulPartitionedCall	input_257dense_1024_18664517dense_1024_18664519*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_dense_1024_layer_call_and_return_conditional_losses_18664286?
"dense_1025/StatefulPartitionedCallStatefulPartitionedCall+dense_1024/StatefulPartitionedCall:output:0dense_1025_18664522dense_1025_18664524*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_dense_1025_layer_call_and_return_conditional_losses_18664303?
"dense_1026/StatefulPartitionedCallStatefulPartitionedCall+dense_1025/StatefulPartitionedCall:output:0dense_1026_18664527dense_1026_18664529*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_dense_1026_layer_call_and_return_conditional_losses_18664320?
"dense_1027/StatefulPartitionedCallStatefulPartitionedCall+dense_1026/StatefulPartitionedCall:output:0dense_1027_18664532dense_1027_18664534*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_dense_1027_layer_call_and_return_conditional_losses_18664337z
IdentityIdentity+dense_1027/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp#^dense_1024/StatefulPartitionedCall#^dense_1025/StatefulPartitionedCall#^dense_1026/StatefulPartitionedCall#^dense_1027/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 2H
"dense_1024/StatefulPartitionedCall"dense_1024/StatefulPartitionedCall2H
"dense_1025/StatefulPartitionedCall"dense_1025/StatefulPartitionedCall2H
"dense_1026/StatefulPartitionedCall"dense_1026/StatefulPartitionedCall2H
"dense_1027/StatefulPartitionedCall"dense_1027/StatefulPartitionedCall:R N
'
_output_shapes
:?????????
#
_user_specified_name	input_257
?
?
G__inference_model_256_layer_call_and_return_conditional_losses_18664344

inputs%
dense_1024_18664287:	!
dense_1024_18664289:	%
dense_1025_18664304:	!
dense_1025_18664306:%
dense_1026_18664321:!
dense_1026_18664323:%
dense_1027_18664338:!
dense_1027_18664340:
identity??"dense_1024/StatefulPartitionedCall?"dense_1025/StatefulPartitionedCall?"dense_1026/StatefulPartitionedCall?"dense_1027/StatefulPartitionedCall?
"dense_1024/StatefulPartitionedCallStatefulPartitionedCallinputsdense_1024_18664287dense_1024_18664289*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_dense_1024_layer_call_and_return_conditional_losses_18664286?
"dense_1025/StatefulPartitionedCallStatefulPartitionedCall+dense_1024/StatefulPartitionedCall:output:0dense_1025_18664304dense_1025_18664306*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_dense_1025_layer_call_and_return_conditional_losses_18664303?
"dense_1026/StatefulPartitionedCallStatefulPartitionedCall+dense_1025/StatefulPartitionedCall:output:0dense_1026_18664321dense_1026_18664323*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_dense_1026_layer_call_and_return_conditional_losses_18664320?
"dense_1027/StatefulPartitionedCallStatefulPartitionedCall+dense_1026/StatefulPartitionedCall:output:0dense_1027_18664338dense_1027_18664340*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_dense_1027_layer_call_and_return_conditional_losses_18664337z
IdentityIdentity+dense_1027/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp#^dense_1024/StatefulPartitionedCall#^dense_1025/StatefulPartitionedCall#^dense_1026/StatefulPartitionedCall#^dense_1027/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 2H
"dense_1024/StatefulPartitionedCall"dense_1024/StatefulPartitionedCall2H
"dense_1025/StatefulPartitionedCall"dense_1025/StatefulPartitionedCall2H
"dense_1026/StatefulPartitionedCall"dense_1026/StatefulPartitionedCall2H
"dense_1027/StatefulPartitionedCall"dense_1027/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?%
?
G__inference_model_256_layer_call_and_return_conditional_losses_18664673

inputs;
)dense_1024_matmul_readvariableop_resource:	8
*dense_1024_biasadd_readvariableop_resource:	;
)dense_1025_matmul_readvariableop_resource:	8
*dense_1025_biasadd_readvariableop_resource:;
)dense_1026_matmul_readvariableop_resource:8
*dense_1026_biasadd_readvariableop_resource:;
)dense_1027_matmul_readvariableop_resource:8
*dense_1027_biasadd_readvariableop_resource:
identity??!dense_1024/BiasAdd/ReadVariableOp? dense_1024/MatMul/ReadVariableOp?!dense_1025/BiasAdd/ReadVariableOp? dense_1025/MatMul/ReadVariableOp?!dense_1026/BiasAdd/ReadVariableOp? dense_1026/MatMul/ReadVariableOp?!dense_1027/BiasAdd/ReadVariableOp? dense_1027/MatMul/ReadVariableOp?
 dense_1024/MatMul/ReadVariableOpReadVariableOp)dense_1024_matmul_readvariableop_resource*
_output_shapes

:	*
dtype0
dense_1024/MatMulMatMulinputs(dense_1024/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	?
!dense_1024/BiasAdd/ReadVariableOpReadVariableOp*dense_1024_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0?
dense_1024/BiasAddBiasAdddense_1024/MatMul:product:0)dense_1024/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	f
dense_1024/ReluReludense_1024/BiasAdd:output:0*
T0*'
_output_shapes
:?????????	?
 dense_1025/MatMul/ReadVariableOpReadVariableOp)dense_1025_matmul_readvariableop_resource*
_output_shapes

:	*
dtype0?
dense_1025/MatMulMatMuldense_1024/Relu:activations:0(dense_1025/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
!dense_1025/BiasAdd/ReadVariableOpReadVariableOp*dense_1025_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_1025/BiasAddBiasAdddense_1025/MatMul:product:0)dense_1025/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f
dense_1025/ReluReludense_1025/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
 dense_1026/MatMul/ReadVariableOpReadVariableOp)dense_1026_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
dense_1026/MatMulMatMuldense_1025/Relu:activations:0(dense_1026/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
!dense_1026/BiasAdd/ReadVariableOpReadVariableOp*dense_1026_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_1026/BiasAddBiasAdddense_1026/MatMul:product:0)dense_1026/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f
dense_1026/ReluReludense_1026/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
 dense_1027/MatMul/ReadVariableOpReadVariableOp)dense_1027_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
dense_1027/MatMulMatMuldense_1026/Relu:activations:0(dense_1027/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
!dense_1027/BiasAdd/ReadVariableOpReadVariableOp*dense_1027_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_1027/BiasAddBiasAdddense_1027/MatMul:product:0)dense_1027/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????l
dense_1027/SigmoidSigmoiddense_1027/BiasAdd:output:0*
T0*'
_output_shapes
:?????????e
IdentityIdentitydense_1027/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp"^dense_1024/BiasAdd/ReadVariableOp!^dense_1024/MatMul/ReadVariableOp"^dense_1025/BiasAdd/ReadVariableOp!^dense_1025/MatMul/ReadVariableOp"^dense_1026/BiasAdd/ReadVariableOp!^dense_1026/MatMul/ReadVariableOp"^dense_1027/BiasAdd/ReadVariableOp!^dense_1027/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 2F
!dense_1024/BiasAdd/ReadVariableOp!dense_1024/BiasAdd/ReadVariableOp2D
 dense_1024/MatMul/ReadVariableOp dense_1024/MatMul/ReadVariableOp2F
!dense_1025/BiasAdd/ReadVariableOp!dense_1025/BiasAdd/ReadVariableOp2D
 dense_1025/MatMul/ReadVariableOp dense_1025/MatMul/ReadVariableOp2F
!dense_1026/BiasAdd/ReadVariableOp!dense_1026/BiasAdd/ReadVariableOp2D
 dense_1026/MatMul/ReadVariableOp dense_1026/MatMul/ReadVariableOp2F
!dense_1027/BiasAdd/ReadVariableOp!dense_1027/BiasAdd/ReadVariableOp2D
 dense_1027/MatMul/ReadVariableOp dense_1027/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
-__inference_dense_1027_layer_call_fn_18664742

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_dense_1027_layer_call_and_return_conditional_losses_18664337o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
G__inference_model_256_layer_call_and_return_conditional_losses_18664514
	input_257%
dense_1024_18664493:	!
dense_1024_18664495:	%
dense_1025_18664498:	!
dense_1025_18664500:%
dense_1026_18664503:!
dense_1026_18664505:%
dense_1027_18664508:!
dense_1027_18664510:
identity??"dense_1024/StatefulPartitionedCall?"dense_1025/StatefulPartitionedCall?"dense_1026/StatefulPartitionedCall?"dense_1027/StatefulPartitionedCall?
"dense_1024/StatefulPartitionedCallStatefulPartitionedCall	input_257dense_1024_18664493dense_1024_18664495*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_dense_1024_layer_call_and_return_conditional_losses_18664286?
"dense_1025/StatefulPartitionedCallStatefulPartitionedCall+dense_1024/StatefulPartitionedCall:output:0dense_1025_18664498dense_1025_18664500*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_dense_1025_layer_call_and_return_conditional_losses_18664303?
"dense_1026/StatefulPartitionedCallStatefulPartitionedCall+dense_1025/StatefulPartitionedCall:output:0dense_1026_18664503dense_1026_18664505*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_dense_1026_layer_call_and_return_conditional_losses_18664320?
"dense_1027/StatefulPartitionedCallStatefulPartitionedCall+dense_1026/StatefulPartitionedCall:output:0dense_1027_18664508dense_1027_18664510*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_dense_1027_layer_call_and_return_conditional_losses_18664337z
IdentityIdentity+dense_1027/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp#^dense_1024/StatefulPartitionedCall#^dense_1025/StatefulPartitionedCall#^dense_1026/StatefulPartitionedCall#^dense_1027/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 2H
"dense_1024/StatefulPartitionedCall"dense_1024/StatefulPartitionedCall2H
"dense_1025/StatefulPartitionedCall"dense_1025/StatefulPartitionedCall2H
"dense_1026/StatefulPartitionedCall"dense_1026/StatefulPartitionedCall2H
"dense_1027/StatefulPartitionedCall"dense_1027/StatefulPartitionedCall:R N
'
_output_shapes
:?????????
#
_user_specified_name	input_257
?

?
H__inference_dense_1026_layer_call_and_return_conditional_losses_18664320

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
H__inference_dense_1027_layer_call_and_return_conditional_losses_18664337

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
H__inference_dense_1025_layer_call_and_return_conditional_losses_18664713

inputs0
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs
?

?
H__inference_dense_1025_layer_call_and_return_conditional_losses_18664303

inputs0
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
?
	input_2572
serving_default_input_257:0?????????>

dense_10270
StatefulPartitionedCall:0?????????tensorflow/serving/predict:?\
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
	optimizer
	variables
trainable_variables
	regularization_losses

	keras_api

signatures
W__call__
*X&call_and_return_all_conditional_losses
Y_default_save_signature"
_tf_keras_network
"
_tf_keras_input_layer
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
Z__call__
*[&call_and_return_all_conditional_losses"
_tf_keras_layer
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
\__call__
*]&call_and_return_all_conditional_losses"
_tf_keras_layer
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
^__call__
*_&call_and_return_all_conditional_losses"
_tf_keras_layer
?

kernel
bias
 	variables
!trainable_variables
"regularization_losses
#	keras_api
`__call__
*a&call_and_return_all_conditional_losses"
_tf_keras_layer
?
$iter

%beta_1

&beta_2
	'decay
(learning_ratemGmHmImJmKmLmMmNvOvPvQvRvSvTvUvV"
	optimizer
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
?
)non_trainable_variables

*layers
+metrics
,layer_regularization_losses
-layer_metrics
	variables
trainable_variables
	regularization_losses
W__call__
Y_default_save_signature
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
_generic_user_object
,
bserving_default"
signature_map
#:!	2dense_1024/kernel
:	2dense_1024/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
.non_trainable_variables

/layers
0metrics
1layer_regularization_losses
2layer_metrics
	variables
trainable_variables
regularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses"
_generic_user_object
#:!	2dense_1025/kernel
:2dense_1025/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
3non_trainable_variables

4layers
5metrics
6layer_regularization_losses
7layer_metrics
	variables
trainable_variables
regularization_losses
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses"
_generic_user_object
#:!2dense_1026/kernel
:2dense_1026/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
8non_trainable_variables

9layers
:metrics
;layer_regularization_losses
<layer_metrics
	variables
trainable_variables
regularization_losses
^__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses"
_generic_user_object
#:!2dense_1027/kernel
:2dense_1027/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
=non_trainable_variables

>layers
?metrics
@layer_regularization_losses
Alayer_metrics
 	variables
!trainable_variables
"regularization_losses
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
'
B0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
N
	Ctotal
	Dcount
E	variables
F	keras_api"
_tf_keras_metric
:  (2total
:  (2count
.
C0
D1"
trackable_list_wrapper
-
E	variables"
_generic_user_object
(:&	2Adam/dense_1024/kernel/m
": 	2Adam/dense_1024/bias/m
(:&	2Adam/dense_1025/kernel/m
": 2Adam/dense_1025/bias/m
(:&2Adam/dense_1026/kernel/m
": 2Adam/dense_1026/bias/m
(:&2Adam/dense_1027/kernel/m
": 2Adam/dense_1027/bias/m
(:&	2Adam/dense_1024/kernel/v
": 	2Adam/dense_1024/bias/v
(:&	2Adam/dense_1025/kernel/v
": 2Adam/dense_1025/bias/v
(:&2Adam/dense_1026/kernel/v
": 2Adam/dense_1026/bias/v
(:&2Adam/dense_1027/kernel/v
": 2Adam/dense_1027/bias/v
?2?
,__inference_model_256_layer_call_fn_18664363
,__inference_model_256_layer_call_fn_18664588
,__inference_model_256_layer_call_fn_18664609
,__inference_model_256_layer_call_fn_18664490?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_model_256_layer_call_and_return_conditional_losses_18664641
G__inference_model_256_layer_call_and_return_conditional_losses_18664673
G__inference_model_256_layer_call_and_return_conditional_losses_18664514
G__inference_model_256_layer_call_and_return_conditional_losses_18664538?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
#__inference__wrapped_model_18664268	input_257"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_dense_1024_layer_call_fn_18664682?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_dense_1024_layer_call_and_return_conditional_losses_18664693?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_dense_1025_layer_call_fn_18664702?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_dense_1025_layer_call_and_return_conditional_losses_18664713?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_dense_1026_layer_call_fn_18664722?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_dense_1026_layer_call_and_return_conditional_losses_18664733?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_dense_1027_layer_call_fn_18664742?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_dense_1027_layer_call_and_return_conditional_losses_18664753?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
&__inference_signature_wrapper_18664567	input_257"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
#__inference__wrapped_model_18664268w2?/
(?%
#? 
	input_257?????????
? "7?4
2

dense_1027$?!

dense_1027??????????
H__inference_dense_1024_layer_call_and_return_conditional_losses_18664693\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????	
? ?
-__inference_dense_1024_layer_call_fn_18664682O/?,
%?"
 ?
inputs?????????
? "??????????	?
H__inference_dense_1025_layer_call_and_return_conditional_losses_18664713\/?,
%?"
 ?
inputs?????????	
? "%?"
?
0?????????
? ?
-__inference_dense_1025_layer_call_fn_18664702O/?,
%?"
 ?
inputs?????????	
? "???????????
H__inference_dense_1026_layer_call_and_return_conditional_losses_18664733\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? ?
-__inference_dense_1026_layer_call_fn_18664722O/?,
%?"
 ?
inputs?????????
? "???????????
H__inference_dense_1027_layer_call_and_return_conditional_losses_18664753\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? ?
-__inference_dense_1027_layer_call_fn_18664742O/?,
%?"
 ?
inputs?????????
? "???????????
G__inference_model_256_layer_call_and_return_conditional_losses_18664514m:?7
0?-
#? 
	input_257?????????
p 

 
? "%?"
?
0?????????
? ?
G__inference_model_256_layer_call_and_return_conditional_losses_18664538m:?7
0?-
#? 
	input_257?????????
p

 
? "%?"
?
0?????????
? ?
G__inference_model_256_layer_call_and_return_conditional_losses_18664641j7?4
-?*
 ?
inputs?????????
p 

 
? "%?"
?
0?????????
? ?
G__inference_model_256_layer_call_and_return_conditional_losses_18664673j7?4
-?*
 ?
inputs?????????
p

 
? "%?"
?
0?????????
? ?
,__inference_model_256_layer_call_fn_18664363`:?7
0?-
#? 
	input_257?????????
p 

 
? "???????????
,__inference_model_256_layer_call_fn_18664490`:?7
0?-
#? 
	input_257?????????
p

 
? "???????????
,__inference_model_256_layer_call_fn_18664588]7?4
-?*
 ?
inputs?????????
p 

 
? "???????????
,__inference_model_256_layer_call_fn_18664609]7?4
-?*
 ?
inputs?????????
p

 
? "???????????
&__inference_signature_wrapper_18664567???<
? 
5?2
0
	input_257#? 
	input_257?????????"7?4
2

dense_1027$?!

dense_1027?????????
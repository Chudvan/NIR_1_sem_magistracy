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
|
dense_688/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_688/kernel
u
$dense_688/kernel/Read/ReadVariableOpReadVariableOpdense_688/kernel*
_output_shapes

:*
dtype0
t
dense_688/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_688/bias
m
"dense_688/bias/Read/ReadVariableOpReadVariableOpdense_688/bias*
_output_shapes
:*
dtype0
|
dense_689/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_689/kernel
u
$dense_689/kernel/Read/ReadVariableOpReadVariableOpdense_689/kernel*
_output_shapes

:*
dtype0
t
dense_689/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_689/bias
m
"dense_689/bias/Read/ReadVariableOpReadVariableOpdense_689/bias*
_output_shapes
:*
dtype0
|
dense_690/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_690/kernel
u
$dense_690/kernel/Read/ReadVariableOpReadVariableOpdense_690/kernel*
_output_shapes

:*
dtype0
t
dense_690/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_690/bias
m
"dense_690/bias/Read/ReadVariableOpReadVariableOpdense_690/bias*
_output_shapes
:*
dtype0
|
dense_691/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_691/kernel
u
$dense_691/kernel/Read/ReadVariableOpReadVariableOpdense_691/kernel*
_output_shapes

:*
dtype0
t
dense_691/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_691/bias
m
"dense_691/bias/Read/ReadVariableOpReadVariableOpdense_691/bias*
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
Adam/dense_688/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_688/kernel/m
?
+Adam/dense_688/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_688/kernel/m*
_output_shapes

:*
dtype0
?
Adam/dense_688/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_688/bias/m
{
)Adam/dense_688/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_688/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_689/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_689/kernel/m
?
+Adam/dense_689/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_689/kernel/m*
_output_shapes

:*
dtype0
?
Adam/dense_689/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_689/bias/m
{
)Adam/dense_689/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_689/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_690/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_690/kernel/m
?
+Adam/dense_690/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_690/kernel/m*
_output_shapes

:*
dtype0
?
Adam/dense_690/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_690/bias/m
{
)Adam/dense_690/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_690/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_691/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_691/kernel/m
?
+Adam/dense_691/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_691/kernel/m*
_output_shapes

:*
dtype0
?
Adam/dense_691/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_691/bias/m
{
)Adam/dense_691/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_691/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_688/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_688/kernel/v
?
+Adam/dense_688/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_688/kernel/v*
_output_shapes

:*
dtype0
?
Adam/dense_688/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_688/bias/v
{
)Adam/dense_688/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_688/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense_689/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_689/kernel/v
?
+Adam/dense_689/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_689/kernel/v*
_output_shapes

:*
dtype0
?
Adam/dense_689/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_689/bias/v
{
)Adam/dense_689/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_689/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense_690/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_690/kernel/v
?
+Adam/dense_690/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_690/kernel/v*
_output_shapes

:*
dtype0
?
Adam/dense_690/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_690/bias/v
{
)Adam/dense_690/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_690/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense_691/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_691/kernel/v
?
+Adam/dense_691/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_691/kernel/v*
_output_shapes

:*
dtype0
?
Adam/dense_691/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_691/bias/v
{
)Adam/dense_691/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_691/bias/v*
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
\Z
VARIABLE_VALUEdense_688/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_688/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
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
\Z
VARIABLE_VALUEdense_689/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_689/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
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
\Z
VARIABLE_VALUEdense_690/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_690/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
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
\Z
VARIABLE_VALUEdense_691/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_691/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
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
}
VARIABLE_VALUEAdam/dense_688/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_688/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_689/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_689/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_690/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_690/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_691/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_691/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_688/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_688/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_689/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_689/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_690/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_690/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_691/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_691/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|
serving_default_input_173Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_173dense_688/kerneldense_688/biasdense_689/kerneldense_689/biasdense_690/kerneldense_690/biasdense_691/kerneldense_691/bias*
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
&__inference_signature_wrapper_18590479
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$dense_688/kernel/Read/ReadVariableOp"dense_688/bias/Read/ReadVariableOp$dense_689/kernel/Read/ReadVariableOp"dense_689/bias/Read/ReadVariableOp$dense_690/kernel/Read/ReadVariableOp"dense_690/bias/Read/ReadVariableOp$dense_691/kernel/Read/ReadVariableOp"dense_691/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp+Adam/dense_688/kernel/m/Read/ReadVariableOp)Adam/dense_688/bias/m/Read/ReadVariableOp+Adam/dense_689/kernel/m/Read/ReadVariableOp)Adam/dense_689/bias/m/Read/ReadVariableOp+Adam/dense_690/kernel/m/Read/ReadVariableOp)Adam/dense_690/bias/m/Read/ReadVariableOp+Adam/dense_691/kernel/m/Read/ReadVariableOp)Adam/dense_691/bias/m/Read/ReadVariableOp+Adam/dense_688/kernel/v/Read/ReadVariableOp)Adam/dense_688/bias/v/Read/ReadVariableOp+Adam/dense_689/kernel/v/Read/ReadVariableOp)Adam/dense_689/bias/v/Read/ReadVariableOp+Adam/dense_690/kernel/v/Read/ReadVariableOp)Adam/dense_690/bias/v/Read/ReadVariableOp+Adam/dense_691/kernel/v/Read/ReadVariableOp)Adam/dense_691/bias/v/Read/ReadVariableOpConst*,
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
!__inference__traced_save_18590781
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_688/kerneldense_688/biasdense_689/kerneldense_689/biasdense_690/kerneldense_690/biasdense_691/kerneldense_691/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/dense_688/kernel/mAdam/dense_688/bias/mAdam/dense_689/kernel/mAdam/dense_689/bias/mAdam/dense_690/kernel/mAdam/dense_690/bias/mAdam/dense_691/kernel/mAdam/dense_691/bias/mAdam/dense_688/kernel/vAdam/dense_688/bias/vAdam/dense_689/kernel/vAdam/dense_689/bias/vAdam/dense_690/kernel/vAdam/dense_690/bias/vAdam/dense_691/kernel/vAdam/dense_691/bias/v*+
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
$__inference__traced_restore_18590884??
?	
?
,__inference_model_172_layer_call_fn_18590521

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
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
G__inference_model_172_layer_call_and_return_conditional_losses_18590362o
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
?
?
,__inference_dense_690_layer_call_fn_18590634

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_690_layer_call_and_return_conditional_losses_18590232o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
G__inference_dense_690_layer_call_and_return_conditional_losses_18590232

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?%
?
G__inference_model_172_layer_call_and_return_conditional_losses_18590553

inputs:
(dense_688_matmul_readvariableop_resource:7
)dense_688_biasadd_readvariableop_resource::
(dense_689_matmul_readvariableop_resource:7
)dense_689_biasadd_readvariableop_resource::
(dense_690_matmul_readvariableop_resource:7
)dense_690_biasadd_readvariableop_resource::
(dense_691_matmul_readvariableop_resource:7
)dense_691_biasadd_readvariableop_resource:
identity?? dense_688/BiasAdd/ReadVariableOp?dense_688/MatMul/ReadVariableOp? dense_689/BiasAdd/ReadVariableOp?dense_689/MatMul/ReadVariableOp? dense_690/BiasAdd/ReadVariableOp?dense_690/MatMul/ReadVariableOp? dense_691/BiasAdd/ReadVariableOp?dense_691/MatMul/ReadVariableOp?
dense_688/MatMul/ReadVariableOpReadVariableOp(dense_688_matmul_readvariableop_resource*
_output_shapes

:*
dtype0}
dense_688/MatMulMatMulinputs'dense_688/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
 dense_688/BiasAdd/ReadVariableOpReadVariableOp)dense_688_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_688/BiasAddBiasAdddense_688/MatMul:product:0(dense_688/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d
dense_688/ReluReludense_688/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
dense_689/MatMul/ReadVariableOpReadVariableOp(dense_689_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
dense_689/MatMulMatMuldense_688/Relu:activations:0'dense_689/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
 dense_689/BiasAdd/ReadVariableOpReadVariableOp)dense_689_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_689/BiasAddBiasAdddense_689/MatMul:product:0(dense_689/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d
dense_689/ReluReludense_689/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
dense_690/MatMul/ReadVariableOpReadVariableOp(dense_690_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
dense_690/MatMulMatMuldense_689/Relu:activations:0'dense_690/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
 dense_690/BiasAdd/ReadVariableOpReadVariableOp)dense_690_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_690/BiasAddBiasAdddense_690/MatMul:product:0(dense_690/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d
dense_690/ReluReludense_690/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
dense_691/MatMul/ReadVariableOpReadVariableOp(dense_691_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
dense_691/MatMulMatMuldense_690/Relu:activations:0'dense_691/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
 dense_691/BiasAdd/ReadVariableOpReadVariableOp)dense_691_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_691/BiasAddBiasAdddense_691/MatMul:product:0(dense_691/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????j
dense_691/SigmoidSigmoiddense_691/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d
IdentityIdentitydense_691/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^dense_688/BiasAdd/ReadVariableOp ^dense_688/MatMul/ReadVariableOp!^dense_689/BiasAdd/ReadVariableOp ^dense_689/MatMul/ReadVariableOp!^dense_690/BiasAdd/ReadVariableOp ^dense_690/MatMul/ReadVariableOp!^dense_691/BiasAdd/ReadVariableOp ^dense_691/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 2D
 dense_688/BiasAdd/ReadVariableOp dense_688/BiasAdd/ReadVariableOp2B
dense_688/MatMul/ReadVariableOpdense_688/MatMul/ReadVariableOp2D
 dense_689/BiasAdd/ReadVariableOp dense_689/BiasAdd/ReadVariableOp2B
dense_689/MatMul/ReadVariableOpdense_689/MatMul/ReadVariableOp2D
 dense_690/BiasAdd/ReadVariableOp dense_690/BiasAdd/ReadVariableOp2B
dense_690/MatMul/ReadVariableOpdense_690/MatMul/ReadVariableOp2D
 dense_691/BiasAdd/ReadVariableOp dense_691/BiasAdd/ReadVariableOp2B
dense_691/MatMul/ReadVariableOpdense_691/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
&__inference_signature_wrapper_18590479
	input_173
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall	input_173unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
#__inference__wrapped_model_18590180o
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
_user_specified_name	input_173
?	
?
,__inference_model_172_layer_call_fn_18590500

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
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
G__inference_model_172_layer_call_and_return_conditional_losses_18590256o
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
?

?
G__inference_dense_691_layer_call_and_return_conditional_losses_18590665

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
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
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?}
?
$__inference__traced_restore_18590884
file_prefix3
!assignvariableop_dense_688_kernel:/
!assignvariableop_1_dense_688_bias:5
#assignvariableop_2_dense_689_kernel:/
!assignvariableop_3_dense_689_bias:5
#assignvariableop_4_dense_690_kernel:/
!assignvariableop_5_dense_690_bias:5
#assignvariableop_6_dense_691_kernel:/
!assignvariableop_7_dense_691_bias:&
assignvariableop_8_adam_iter:	 (
assignvariableop_9_adam_beta_1: )
assignvariableop_10_adam_beta_2: (
assignvariableop_11_adam_decay: 0
&assignvariableop_12_adam_learning_rate: #
assignvariableop_13_total: #
assignvariableop_14_count: =
+assignvariableop_15_adam_dense_688_kernel_m:7
)assignvariableop_16_adam_dense_688_bias_m:=
+assignvariableop_17_adam_dense_689_kernel_m:7
)assignvariableop_18_adam_dense_689_bias_m:=
+assignvariableop_19_adam_dense_690_kernel_m:7
)assignvariableop_20_adam_dense_690_bias_m:=
+assignvariableop_21_adam_dense_691_kernel_m:7
)assignvariableop_22_adam_dense_691_bias_m:=
+assignvariableop_23_adam_dense_688_kernel_v:7
)assignvariableop_24_adam_dense_688_bias_v:=
+assignvariableop_25_adam_dense_689_kernel_v:7
)assignvariableop_26_adam_dense_689_bias_v:=
+assignvariableop_27_adam_dense_690_kernel_v:7
)assignvariableop_28_adam_dense_690_bias_v:=
+assignvariableop_29_adam_dense_691_kernel_v:7
)assignvariableop_30_adam_dense_691_bias_v:
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
AssignVariableOpAssignVariableOp!assignvariableop_dense_688_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp!assignvariableop_1_dense_688_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp#assignvariableop_2_dense_689_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp!assignvariableop_3_dense_689_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp#assignvariableop_4_dense_690_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp!assignvariableop_5_dense_690_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp#assignvariableop_6_dense_691_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp!assignvariableop_7_dense_691_biasIdentity_7:output:0"/device:CPU:0*
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
AssignVariableOp_15AssignVariableOp+assignvariableop_15_adam_dense_688_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp)assignvariableop_16_adam_dense_688_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp+assignvariableop_17_adam_dense_689_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp)assignvariableop_18_adam_dense_689_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp+assignvariableop_19_adam_dense_690_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp)assignvariableop_20_adam_dense_690_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp+assignvariableop_21_adam_dense_691_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp)assignvariableop_22_adam_dense_691_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp+assignvariableop_23_adam_dense_688_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp)assignvariableop_24_adam_dense_688_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp+assignvariableop_25_adam_dense_689_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOp)assignvariableop_26_adam_dense_689_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOp+assignvariableop_27_adam_dense_690_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOp)assignvariableop_28_adam_dense_690_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOp+assignvariableop_29_adam_dense_691_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOp)assignvariableop_30_adam_dense_691_bias_vIdentity_30:output:0"/device:CPU:0*
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

?
G__inference_dense_690_layer_call_and_return_conditional_losses_18590645

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
,__inference_model_172_layer_call_fn_18590402
	input_173
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall	input_173unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
G__inference_model_172_layer_call_and_return_conditional_losses_18590362o
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
_user_specified_name	input_173
?
?
G__inference_model_172_layer_call_and_return_conditional_losses_18590450
	input_173$
dense_688_18590429: 
dense_688_18590431:$
dense_689_18590434: 
dense_689_18590436:$
dense_690_18590439: 
dense_690_18590441:$
dense_691_18590444: 
dense_691_18590446:
identity??!dense_688/StatefulPartitionedCall?!dense_689/StatefulPartitionedCall?!dense_690/StatefulPartitionedCall?!dense_691/StatefulPartitionedCall?
!dense_688/StatefulPartitionedCallStatefulPartitionedCall	input_173dense_688_18590429dense_688_18590431*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_688_layer_call_and_return_conditional_losses_18590198?
!dense_689/StatefulPartitionedCallStatefulPartitionedCall*dense_688/StatefulPartitionedCall:output:0dense_689_18590434dense_689_18590436*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_689_layer_call_and_return_conditional_losses_18590215?
!dense_690/StatefulPartitionedCallStatefulPartitionedCall*dense_689/StatefulPartitionedCall:output:0dense_690_18590439dense_690_18590441*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_690_layer_call_and_return_conditional_losses_18590232?
!dense_691/StatefulPartitionedCallStatefulPartitionedCall*dense_690/StatefulPartitionedCall:output:0dense_691_18590444dense_691_18590446*
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
GPU 2J 8? *P
fKRI
G__inference_dense_691_layer_call_and_return_conditional_losses_18590249y
IdentityIdentity*dense_691/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp"^dense_688/StatefulPartitionedCall"^dense_689/StatefulPartitionedCall"^dense_690/StatefulPartitionedCall"^dense_691/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 2F
!dense_688/StatefulPartitionedCall!dense_688/StatefulPartitionedCall2F
!dense_689/StatefulPartitionedCall!dense_689/StatefulPartitionedCall2F
!dense_690/StatefulPartitionedCall!dense_690/StatefulPartitionedCall2F
!dense_691/StatefulPartitionedCall!dense_691/StatefulPartitionedCall:R N
'
_output_shapes
:?????????
#
_user_specified_name	input_173
?C
?
!__inference__traced_save_18590781
file_prefix/
+savev2_dense_688_kernel_read_readvariableop-
)savev2_dense_688_bias_read_readvariableop/
+savev2_dense_689_kernel_read_readvariableop-
)savev2_dense_689_bias_read_readvariableop/
+savev2_dense_690_kernel_read_readvariableop-
)savev2_dense_690_bias_read_readvariableop/
+savev2_dense_691_kernel_read_readvariableop-
)savev2_dense_691_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop6
2savev2_adam_dense_688_kernel_m_read_readvariableop4
0savev2_adam_dense_688_bias_m_read_readvariableop6
2savev2_adam_dense_689_kernel_m_read_readvariableop4
0savev2_adam_dense_689_bias_m_read_readvariableop6
2savev2_adam_dense_690_kernel_m_read_readvariableop4
0savev2_adam_dense_690_bias_m_read_readvariableop6
2savev2_adam_dense_691_kernel_m_read_readvariableop4
0savev2_adam_dense_691_bias_m_read_readvariableop6
2savev2_adam_dense_688_kernel_v_read_readvariableop4
0savev2_adam_dense_688_bias_v_read_readvariableop6
2savev2_adam_dense_689_kernel_v_read_readvariableop4
0savev2_adam_dense_689_bias_v_read_readvariableop6
2savev2_adam_dense_690_kernel_v_read_readvariableop4
0savev2_adam_dense_690_bias_v_read_readvariableop6
2savev2_adam_dense_691_kernel_v_read_readvariableop4
0savev2_adam_dense_691_bias_v_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_dense_688_kernel_read_readvariableop)savev2_dense_688_bias_read_readvariableop+savev2_dense_689_kernel_read_readvariableop)savev2_dense_689_bias_read_readvariableop+savev2_dense_690_kernel_read_readvariableop)savev2_dense_690_bias_read_readvariableop+savev2_dense_691_kernel_read_readvariableop)savev2_dense_691_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop2savev2_adam_dense_688_kernel_m_read_readvariableop0savev2_adam_dense_688_bias_m_read_readvariableop2savev2_adam_dense_689_kernel_m_read_readvariableop0savev2_adam_dense_689_bias_m_read_readvariableop2savev2_adam_dense_690_kernel_m_read_readvariableop0savev2_adam_dense_690_bias_m_read_readvariableop2savev2_adam_dense_691_kernel_m_read_readvariableop0savev2_adam_dense_691_bias_m_read_readvariableop2savev2_adam_dense_688_kernel_v_read_readvariableop0savev2_adam_dense_688_bias_v_read_readvariableop2savev2_adam_dense_689_kernel_v_read_readvariableop0savev2_adam_dense_689_bias_v_read_readvariableop2savev2_adam_dense_690_kernel_v_read_readvariableop0savev2_adam_dense_690_bias_v_read_readvariableop2savev2_adam_dense_691_kernel_v_read_readvariableop0savev2_adam_dense_691_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
?: ::::::::: : : : : : : ::::::::::::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 
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

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
:: 

_output_shapes
: 
?

?
G__inference_dense_691_layer_call_and_return_conditional_losses_18590249

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
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
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
G__inference_model_172_layer_call_and_return_conditional_losses_18590362

inputs$
dense_688_18590341: 
dense_688_18590343:$
dense_689_18590346: 
dense_689_18590348:$
dense_690_18590351: 
dense_690_18590353:$
dense_691_18590356: 
dense_691_18590358:
identity??!dense_688/StatefulPartitionedCall?!dense_689/StatefulPartitionedCall?!dense_690/StatefulPartitionedCall?!dense_691/StatefulPartitionedCall?
!dense_688/StatefulPartitionedCallStatefulPartitionedCallinputsdense_688_18590341dense_688_18590343*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_688_layer_call_and_return_conditional_losses_18590198?
!dense_689/StatefulPartitionedCallStatefulPartitionedCall*dense_688/StatefulPartitionedCall:output:0dense_689_18590346dense_689_18590348*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_689_layer_call_and_return_conditional_losses_18590215?
!dense_690/StatefulPartitionedCallStatefulPartitionedCall*dense_689/StatefulPartitionedCall:output:0dense_690_18590351dense_690_18590353*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_690_layer_call_and_return_conditional_losses_18590232?
!dense_691/StatefulPartitionedCallStatefulPartitionedCall*dense_690/StatefulPartitionedCall:output:0dense_691_18590356dense_691_18590358*
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
GPU 2J 8? *P
fKRI
G__inference_dense_691_layer_call_and_return_conditional_losses_18590249y
IdentityIdentity*dense_691/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp"^dense_688/StatefulPartitionedCall"^dense_689/StatefulPartitionedCall"^dense_690/StatefulPartitionedCall"^dense_691/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 2F
!dense_688/StatefulPartitionedCall!dense_688/StatefulPartitionedCall2F
!dense_689/StatefulPartitionedCall!dense_689/StatefulPartitionedCall2F
!dense_690/StatefulPartitionedCall!dense_690/StatefulPartitionedCall2F
!dense_691/StatefulPartitionedCall!dense_691/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
G__inference_dense_688_layer_call_and_return_conditional_losses_18590198

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????w
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
,__inference_dense_689_layer_call_fn_18590614

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_689_layer_call_and_return_conditional_losses_18590215o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?+
?
#__inference__wrapped_model_18590180
	input_173D
2model_172_dense_688_matmul_readvariableop_resource:A
3model_172_dense_688_biasadd_readvariableop_resource:D
2model_172_dense_689_matmul_readvariableop_resource:A
3model_172_dense_689_biasadd_readvariableop_resource:D
2model_172_dense_690_matmul_readvariableop_resource:A
3model_172_dense_690_biasadd_readvariableop_resource:D
2model_172_dense_691_matmul_readvariableop_resource:A
3model_172_dense_691_biasadd_readvariableop_resource:
identity??*model_172/dense_688/BiasAdd/ReadVariableOp?)model_172/dense_688/MatMul/ReadVariableOp?*model_172/dense_689/BiasAdd/ReadVariableOp?)model_172/dense_689/MatMul/ReadVariableOp?*model_172/dense_690/BiasAdd/ReadVariableOp?)model_172/dense_690/MatMul/ReadVariableOp?*model_172/dense_691/BiasAdd/ReadVariableOp?)model_172/dense_691/MatMul/ReadVariableOp?
)model_172/dense_688/MatMul/ReadVariableOpReadVariableOp2model_172_dense_688_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
model_172/dense_688/MatMulMatMul	input_1731model_172/dense_688/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
*model_172/dense_688/BiasAdd/ReadVariableOpReadVariableOp3model_172_dense_688_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model_172/dense_688/BiasAddBiasAdd$model_172/dense_688/MatMul:product:02model_172/dense_688/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????x
model_172/dense_688/ReluRelu$model_172/dense_688/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
)model_172/dense_689/MatMul/ReadVariableOpReadVariableOp2model_172_dense_689_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
model_172/dense_689/MatMulMatMul&model_172/dense_688/Relu:activations:01model_172/dense_689/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
*model_172/dense_689/BiasAdd/ReadVariableOpReadVariableOp3model_172_dense_689_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model_172/dense_689/BiasAddBiasAdd$model_172/dense_689/MatMul:product:02model_172/dense_689/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????x
model_172/dense_689/ReluRelu$model_172/dense_689/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
)model_172/dense_690/MatMul/ReadVariableOpReadVariableOp2model_172_dense_690_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
model_172/dense_690/MatMulMatMul&model_172/dense_689/Relu:activations:01model_172/dense_690/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
*model_172/dense_690/BiasAdd/ReadVariableOpReadVariableOp3model_172_dense_690_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model_172/dense_690/BiasAddBiasAdd$model_172/dense_690/MatMul:product:02model_172/dense_690/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????x
model_172/dense_690/ReluRelu$model_172/dense_690/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
)model_172/dense_691/MatMul/ReadVariableOpReadVariableOp2model_172_dense_691_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
model_172/dense_691/MatMulMatMul&model_172/dense_690/Relu:activations:01model_172/dense_691/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
*model_172/dense_691/BiasAdd/ReadVariableOpReadVariableOp3model_172_dense_691_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model_172/dense_691/BiasAddBiasAdd$model_172/dense_691/MatMul:product:02model_172/dense_691/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????~
model_172/dense_691/SigmoidSigmoid$model_172/dense_691/BiasAdd:output:0*
T0*'
_output_shapes
:?????????n
IdentityIdentitymodel_172/dense_691/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp+^model_172/dense_688/BiasAdd/ReadVariableOp*^model_172/dense_688/MatMul/ReadVariableOp+^model_172/dense_689/BiasAdd/ReadVariableOp*^model_172/dense_689/MatMul/ReadVariableOp+^model_172/dense_690/BiasAdd/ReadVariableOp*^model_172/dense_690/MatMul/ReadVariableOp+^model_172/dense_691/BiasAdd/ReadVariableOp*^model_172/dense_691/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 2X
*model_172/dense_688/BiasAdd/ReadVariableOp*model_172/dense_688/BiasAdd/ReadVariableOp2V
)model_172/dense_688/MatMul/ReadVariableOp)model_172/dense_688/MatMul/ReadVariableOp2X
*model_172/dense_689/BiasAdd/ReadVariableOp*model_172/dense_689/BiasAdd/ReadVariableOp2V
)model_172/dense_689/MatMul/ReadVariableOp)model_172/dense_689/MatMul/ReadVariableOp2X
*model_172/dense_690/BiasAdd/ReadVariableOp*model_172/dense_690/BiasAdd/ReadVariableOp2V
)model_172/dense_690/MatMul/ReadVariableOp)model_172/dense_690/MatMul/ReadVariableOp2X
*model_172/dense_691/BiasAdd/ReadVariableOp*model_172/dense_691/BiasAdd/ReadVariableOp2V
)model_172/dense_691/MatMul/ReadVariableOp)model_172/dense_691/MatMul/ReadVariableOp:R N
'
_output_shapes
:?????????
#
_user_specified_name	input_173
?	
?
,__inference_model_172_layer_call_fn_18590275
	input_173
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall	input_173unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
G__inference_model_172_layer_call_and_return_conditional_losses_18590256o
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
_user_specified_name	input_173
?

?
G__inference_dense_688_layer_call_and_return_conditional_losses_18590605

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????w
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
,__inference_dense_688_layer_call_fn_18590594

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_688_layer_call_and_return_conditional_losses_18590198o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
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
G__inference_dense_689_layer_call_and_return_conditional_losses_18590625

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
G__inference_model_172_layer_call_and_return_conditional_losses_18590256

inputs$
dense_688_18590199: 
dense_688_18590201:$
dense_689_18590216: 
dense_689_18590218:$
dense_690_18590233: 
dense_690_18590235:$
dense_691_18590250: 
dense_691_18590252:
identity??!dense_688/StatefulPartitionedCall?!dense_689/StatefulPartitionedCall?!dense_690/StatefulPartitionedCall?!dense_691/StatefulPartitionedCall?
!dense_688/StatefulPartitionedCallStatefulPartitionedCallinputsdense_688_18590199dense_688_18590201*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_688_layer_call_and_return_conditional_losses_18590198?
!dense_689/StatefulPartitionedCallStatefulPartitionedCall*dense_688/StatefulPartitionedCall:output:0dense_689_18590216dense_689_18590218*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_689_layer_call_and_return_conditional_losses_18590215?
!dense_690/StatefulPartitionedCallStatefulPartitionedCall*dense_689/StatefulPartitionedCall:output:0dense_690_18590233dense_690_18590235*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_690_layer_call_and_return_conditional_losses_18590232?
!dense_691/StatefulPartitionedCallStatefulPartitionedCall*dense_690/StatefulPartitionedCall:output:0dense_691_18590250dense_691_18590252*
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
GPU 2J 8? *P
fKRI
G__inference_dense_691_layer_call_and_return_conditional_losses_18590249y
IdentityIdentity*dense_691/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp"^dense_688/StatefulPartitionedCall"^dense_689/StatefulPartitionedCall"^dense_690/StatefulPartitionedCall"^dense_691/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 2F
!dense_688/StatefulPartitionedCall!dense_688/StatefulPartitionedCall2F
!dense_689/StatefulPartitionedCall!dense_689/StatefulPartitionedCall2F
!dense_690/StatefulPartitionedCall!dense_690/StatefulPartitionedCall2F
!dense_691/StatefulPartitionedCall!dense_691/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?%
?
G__inference_model_172_layer_call_and_return_conditional_losses_18590585

inputs:
(dense_688_matmul_readvariableop_resource:7
)dense_688_biasadd_readvariableop_resource::
(dense_689_matmul_readvariableop_resource:7
)dense_689_biasadd_readvariableop_resource::
(dense_690_matmul_readvariableop_resource:7
)dense_690_biasadd_readvariableop_resource::
(dense_691_matmul_readvariableop_resource:7
)dense_691_biasadd_readvariableop_resource:
identity?? dense_688/BiasAdd/ReadVariableOp?dense_688/MatMul/ReadVariableOp? dense_689/BiasAdd/ReadVariableOp?dense_689/MatMul/ReadVariableOp? dense_690/BiasAdd/ReadVariableOp?dense_690/MatMul/ReadVariableOp? dense_691/BiasAdd/ReadVariableOp?dense_691/MatMul/ReadVariableOp?
dense_688/MatMul/ReadVariableOpReadVariableOp(dense_688_matmul_readvariableop_resource*
_output_shapes

:*
dtype0}
dense_688/MatMulMatMulinputs'dense_688/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
 dense_688/BiasAdd/ReadVariableOpReadVariableOp)dense_688_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_688/BiasAddBiasAdddense_688/MatMul:product:0(dense_688/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d
dense_688/ReluReludense_688/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
dense_689/MatMul/ReadVariableOpReadVariableOp(dense_689_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
dense_689/MatMulMatMuldense_688/Relu:activations:0'dense_689/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
 dense_689/BiasAdd/ReadVariableOpReadVariableOp)dense_689_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_689/BiasAddBiasAdddense_689/MatMul:product:0(dense_689/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d
dense_689/ReluReludense_689/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
dense_690/MatMul/ReadVariableOpReadVariableOp(dense_690_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
dense_690/MatMulMatMuldense_689/Relu:activations:0'dense_690/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
 dense_690/BiasAdd/ReadVariableOpReadVariableOp)dense_690_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_690/BiasAddBiasAdddense_690/MatMul:product:0(dense_690/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d
dense_690/ReluReludense_690/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
dense_691/MatMul/ReadVariableOpReadVariableOp(dense_691_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
dense_691/MatMulMatMuldense_690/Relu:activations:0'dense_691/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
 dense_691/BiasAdd/ReadVariableOpReadVariableOp)dense_691_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_691/BiasAddBiasAdddense_691/MatMul:product:0(dense_691/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????j
dense_691/SigmoidSigmoiddense_691/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d
IdentityIdentitydense_691/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^dense_688/BiasAdd/ReadVariableOp ^dense_688/MatMul/ReadVariableOp!^dense_689/BiasAdd/ReadVariableOp ^dense_689/MatMul/ReadVariableOp!^dense_690/BiasAdd/ReadVariableOp ^dense_690/MatMul/ReadVariableOp!^dense_691/BiasAdd/ReadVariableOp ^dense_691/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 2D
 dense_688/BiasAdd/ReadVariableOp dense_688/BiasAdd/ReadVariableOp2B
dense_688/MatMul/ReadVariableOpdense_688/MatMul/ReadVariableOp2D
 dense_689/BiasAdd/ReadVariableOp dense_689/BiasAdd/ReadVariableOp2B
dense_689/MatMul/ReadVariableOpdense_689/MatMul/ReadVariableOp2D
 dense_690/BiasAdd/ReadVariableOp dense_690/BiasAdd/ReadVariableOp2B
dense_690/MatMul/ReadVariableOpdense_690/MatMul/ReadVariableOp2D
 dense_691/BiasAdd/ReadVariableOp dense_691/BiasAdd/ReadVariableOp2B
dense_691/MatMul/ReadVariableOpdense_691/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
,__inference_dense_691_layer_call_fn_18590654

inputs
unknown:
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
GPU 2J 8? *P
fKRI
G__inference_dense_691_layer_call_and_return_conditional_losses_18590249o
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
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
G__inference_dense_689_layer_call_and_return_conditional_losses_18590215

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
G__inference_model_172_layer_call_and_return_conditional_losses_18590426
	input_173$
dense_688_18590405: 
dense_688_18590407:$
dense_689_18590410: 
dense_689_18590412:$
dense_690_18590415: 
dense_690_18590417:$
dense_691_18590420: 
dense_691_18590422:
identity??!dense_688/StatefulPartitionedCall?!dense_689/StatefulPartitionedCall?!dense_690/StatefulPartitionedCall?!dense_691/StatefulPartitionedCall?
!dense_688/StatefulPartitionedCallStatefulPartitionedCall	input_173dense_688_18590405dense_688_18590407*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_688_layer_call_and_return_conditional_losses_18590198?
!dense_689/StatefulPartitionedCallStatefulPartitionedCall*dense_688/StatefulPartitionedCall:output:0dense_689_18590410dense_689_18590412*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_689_layer_call_and_return_conditional_losses_18590215?
!dense_690/StatefulPartitionedCallStatefulPartitionedCall*dense_689/StatefulPartitionedCall:output:0dense_690_18590415dense_690_18590417*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_690_layer_call_and_return_conditional_losses_18590232?
!dense_691/StatefulPartitionedCallStatefulPartitionedCall*dense_690/StatefulPartitionedCall:output:0dense_691_18590420dense_691_18590422*
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
GPU 2J 8? *P
fKRI
G__inference_dense_691_layer_call_and_return_conditional_losses_18590249y
IdentityIdentity*dense_691/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp"^dense_688/StatefulPartitionedCall"^dense_689/StatefulPartitionedCall"^dense_690/StatefulPartitionedCall"^dense_691/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 2F
!dense_688/StatefulPartitionedCall!dense_688/StatefulPartitionedCall2F
!dense_689/StatefulPartitionedCall!dense_689/StatefulPartitionedCall2F
!dense_690/StatefulPartitionedCall!dense_690/StatefulPartitionedCall2F
!dense_691/StatefulPartitionedCall!dense_691/StatefulPartitionedCall:R N
'
_output_shapes
:?????????
#
_user_specified_name	input_173"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
?
	input_1732
serving_default_input_173:0?????????=
	dense_6910
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
": 2dense_688/kernel
:2dense_688/bias
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
": 2dense_689/kernel
:2dense_689/bias
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
": 2dense_690/kernel
:2dense_690/bias
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
": 2dense_691/kernel
:2dense_691/bias
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
':%2Adam/dense_688/kernel/m
!:2Adam/dense_688/bias/m
':%2Adam/dense_689/kernel/m
!:2Adam/dense_689/bias/m
':%2Adam/dense_690/kernel/m
!:2Adam/dense_690/bias/m
':%2Adam/dense_691/kernel/m
!:2Adam/dense_691/bias/m
':%2Adam/dense_688/kernel/v
!:2Adam/dense_688/bias/v
':%2Adam/dense_689/kernel/v
!:2Adam/dense_689/bias/v
':%2Adam/dense_690/kernel/v
!:2Adam/dense_690/bias/v
':%2Adam/dense_691/kernel/v
!:2Adam/dense_691/bias/v
?2?
,__inference_model_172_layer_call_fn_18590275
,__inference_model_172_layer_call_fn_18590500
,__inference_model_172_layer_call_fn_18590521
,__inference_model_172_layer_call_fn_18590402?
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
G__inference_model_172_layer_call_and_return_conditional_losses_18590553
G__inference_model_172_layer_call_and_return_conditional_losses_18590585
G__inference_model_172_layer_call_and_return_conditional_losses_18590426
G__inference_model_172_layer_call_and_return_conditional_losses_18590450?
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
#__inference__wrapped_model_18590180	input_173"?
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
,__inference_dense_688_layer_call_fn_18590594?
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
G__inference_dense_688_layer_call_and_return_conditional_losses_18590605?
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
,__inference_dense_689_layer_call_fn_18590614?
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
G__inference_dense_689_layer_call_and_return_conditional_losses_18590625?
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
,__inference_dense_690_layer_call_fn_18590634?
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
G__inference_dense_690_layer_call_and_return_conditional_losses_18590645?
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
,__inference_dense_691_layer_call_fn_18590654?
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
G__inference_dense_691_layer_call_and_return_conditional_losses_18590665?
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
&__inference_signature_wrapper_18590479	input_173"?
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
#__inference__wrapped_model_18590180u2?/
(?%
#? 
	input_173?????????
? "5?2
0
	dense_691#? 
	dense_691??????????
G__inference_dense_688_layer_call_and_return_conditional_losses_18590605\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? 
,__inference_dense_688_layer_call_fn_18590594O/?,
%?"
 ?
inputs?????????
? "???????????
G__inference_dense_689_layer_call_and_return_conditional_losses_18590625\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? 
,__inference_dense_689_layer_call_fn_18590614O/?,
%?"
 ?
inputs?????????
? "???????????
G__inference_dense_690_layer_call_and_return_conditional_losses_18590645\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? 
,__inference_dense_690_layer_call_fn_18590634O/?,
%?"
 ?
inputs?????????
? "???????????
G__inference_dense_691_layer_call_and_return_conditional_losses_18590665\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? 
,__inference_dense_691_layer_call_fn_18590654O/?,
%?"
 ?
inputs?????????
? "???????????
G__inference_model_172_layer_call_and_return_conditional_losses_18590426m:?7
0?-
#? 
	input_173?????????
p 

 
? "%?"
?
0?????????
? ?
G__inference_model_172_layer_call_and_return_conditional_losses_18590450m:?7
0?-
#? 
	input_173?????????
p

 
? "%?"
?
0?????????
? ?
G__inference_model_172_layer_call_and_return_conditional_losses_18590553j7?4
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
G__inference_model_172_layer_call_and_return_conditional_losses_18590585j7?4
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
,__inference_model_172_layer_call_fn_18590275`:?7
0?-
#? 
	input_173?????????
p 

 
? "???????????
,__inference_model_172_layer_call_fn_18590402`:?7
0?-
#? 
	input_173?????????
p

 
? "???????????
,__inference_model_172_layer_call_fn_18590500]7?4
-?*
 ?
inputs?????????
p 

 
? "???????????
,__inference_model_172_layer_call_fn_18590521]7?4
-?*
 ?
inputs?????????
p

 
? "???????????
&__inference_signature_wrapper_18590479???<
? 
5?2
0
	input_173#? 
	input_173?????????"5?2
0
	dense_691#? 
	dense_691?????????
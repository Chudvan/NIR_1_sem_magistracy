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
dense_840/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_840/kernel
u
$dense_840/kernel/Read/ReadVariableOpReadVariableOpdense_840/kernel*
_output_shapes

:*
dtype0
t
dense_840/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_840/bias
m
"dense_840/bias/Read/ReadVariableOpReadVariableOpdense_840/bias*
_output_shapes
:*
dtype0
|
dense_841/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*!
shared_namedense_841/kernel
u
$dense_841/kernel/Read/ReadVariableOpReadVariableOpdense_841/kernel*
_output_shapes

:
*
dtype0
t
dense_841/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_841/bias
m
"dense_841/bias/Read/ReadVariableOpReadVariableOpdense_841/bias*
_output_shapes
:
*
dtype0
|
dense_842/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*!
shared_namedense_842/kernel
u
$dense_842/kernel/Read/ReadVariableOpReadVariableOpdense_842/kernel*
_output_shapes

:
*
dtype0
t
dense_842/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_842/bias
m
"dense_842/bias/Read/ReadVariableOpReadVariableOpdense_842/bias*
_output_shapes
:*
dtype0
|
dense_843/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_843/kernel
u
$dense_843/kernel/Read/ReadVariableOpReadVariableOpdense_843/kernel*
_output_shapes

:*
dtype0
t
dense_843/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_843/bias
m
"dense_843/bias/Read/ReadVariableOpReadVariableOpdense_843/bias*
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
Adam/dense_840/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_840/kernel/m
?
+Adam/dense_840/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_840/kernel/m*
_output_shapes

:*
dtype0
?
Adam/dense_840/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_840/bias/m
{
)Adam/dense_840/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_840/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_841/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*(
shared_nameAdam/dense_841/kernel/m
?
+Adam/dense_841/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_841/kernel/m*
_output_shapes

:
*
dtype0
?
Adam/dense_841/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/dense_841/bias/m
{
)Adam/dense_841/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_841/bias/m*
_output_shapes
:
*
dtype0
?
Adam/dense_842/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*(
shared_nameAdam/dense_842/kernel/m
?
+Adam/dense_842/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_842/kernel/m*
_output_shapes

:
*
dtype0
?
Adam/dense_842/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_842/bias/m
{
)Adam/dense_842/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_842/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_843/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_843/kernel/m
?
+Adam/dense_843/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_843/kernel/m*
_output_shapes

:*
dtype0
?
Adam/dense_843/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_843/bias/m
{
)Adam/dense_843/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_843/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_840/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_840/kernel/v
?
+Adam/dense_840/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_840/kernel/v*
_output_shapes

:*
dtype0
?
Adam/dense_840/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_840/bias/v
{
)Adam/dense_840/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_840/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense_841/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*(
shared_nameAdam/dense_841/kernel/v
?
+Adam/dense_841/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_841/kernel/v*
_output_shapes

:
*
dtype0
?
Adam/dense_841/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/dense_841/bias/v
{
)Adam/dense_841/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_841/bias/v*
_output_shapes
:
*
dtype0
?
Adam/dense_842/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*(
shared_nameAdam/dense_842/kernel/v
?
+Adam/dense_842/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_842/kernel/v*
_output_shapes

:
*
dtype0
?
Adam/dense_842/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_842/bias/v
{
)Adam/dense_842/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_842/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense_843/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_843/kernel/v
?
+Adam/dense_843/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_843/kernel/v*
_output_shapes

:*
dtype0
?
Adam/dense_843/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_843/bias/v
{
)Adam/dense_843/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_843/bias/v*
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
VARIABLE_VALUEdense_840/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_840/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEdense_841/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_841/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEdense_842/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_842/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEdense_843/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_843/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEAdam/dense_840/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_840/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_841/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_841/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_842/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_842/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_843/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_843/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_840/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_840/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_841/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_841/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_842/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_842/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_843/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_843/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|
serving_default_input_211Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_211dense_840/kerneldense_840/biasdense_841/kerneldense_841/biasdense_842/kerneldense_842/biasdense_843/kerneldense_843/bias*
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
&__inference_signature_wrapper_18623995
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$dense_840/kernel/Read/ReadVariableOp"dense_840/bias/Read/ReadVariableOp$dense_841/kernel/Read/ReadVariableOp"dense_841/bias/Read/ReadVariableOp$dense_842/kernel/Read/ReadVariableOp"dense_842/bias/Read/ReadVariableOp$dense_843/kernel/Read/ReadVariableOp"dense_843/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp+Adam/dense_840/kernel/m/Read/ReadVariableOp)Adam/dense_840/bias/m/Read/ReadVariableOp+Adam/dense_841/kernel/m/Read/ReadVariableOp)Adam/dense_841/bias/m/Read/ReadVariableOp+Adam/dense_842/kernel/m/Read/ReadVariableOp)Adam/dense_842/bias/m/Read/ReadVariableOp+Adam/dense_843/kernel/m/Read/ReadVariableOp)Adam/dense_843/bias/m/Read/ReadVariableOp+Adam/dense_840/kernel/v/Read/ReadVariableOp)Adam/dense_840/bias/v/Read/ReadVariableOp+Adam/dense_841/kernel/v/Read/ReadVariableOp)Adam/dense_841/bias/v/Read/ReadVariableOp+Adam/dense_842/kernel/v/Read/ReadVariableOp)Adam/dense_842/bias/v/Read/ReadVariableOp+Adam/dense_843/kernel/v/Read/ReadVariableOp)Adam/dense_843/bias/v/Read/ReadVariableOpConst*,
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
!__inference__traced_save_18624297
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_840/kerneldense_840/biasdense_841/kerneldense_841/biasdense_842/kerneldense_842/biasdense_843/kerneldense_843/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/dense_840/kernel/mAdam/dense_840/bias/mAdam/dense_841/kernel/mAdam/dense_841/bias/mAdam/dense_842/kernel/mAdam/dense_842/bias/mAdam/dense_843/kernel/mAdam/dense_843/bias/mAdam/dense_840/kernel/vAdam/dense_840/bias/vAdam/dense_841/kernel/vAdam/dense_841/bias/vAdam/dense_842/kernel/vAdam/dense_842/bias/vAdam/dense_843/kernel/vAdam/dense_843/bias/v*+
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
$__inference__traced_restore_18624400??
?

?
G__inference_dense_841_layer_call_and_return_conditional_losses_18623731

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????
a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????
w
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
G__inference_model_210_layer_call_and_return_conditional_losses_18623878

inputs$
dense_840_18623857: 
dense_840_18623859:$
dense_841_18623862:
 
dense_841_18623864:
$
dense_842_18623867:
 
dense_842_18623869:$
dense_843_18623872: 
dense_843_18623874:
identity??!dense_840/StatefulPartitionedCall?!dense_841/StatefulPartitionedCall?!dense_842/StatefulPartitionedCall?!dense_843/StatefulPartitionedCall?
!dense_840/StatefulPartitionedCallStatefulPartitionedCallinputsdense_840_18623857dense_840_18623859*
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
G__inference_dense_840_layer_call_and_return_conditional_losses_18623714?
!dense_841/StatefulPartitionedCallStatefulPartitionedCall*dense_840/StatefulPartitionedCall:output:0dense_841_18623862dense_841_18623864*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_841_layer_call_and_return_conditional_losses_18623731?
!dense_842/StatefulPartitionedCallStatefulPartitionedCall*dense_841/StatefulPartitionedCall:output:0dense_842_18623867dense_842_18623869*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_842_layer_call_and_return_conditional_losses_18623748?
!dense_843/StatefulPartitionedCallStatefulPartitionedCall*dense_842/StatefulPartitionedCall:output:0dense_843_18623872dense_843_18623874*
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
G__inference_dense_843_layer_call_and_return_conditional_losses_18623765y
IdentityIdentity*dense_843/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp"^dense_840/StatefulPartitionedCall"^dense_841/StatefulPartitionedCall"^dense_842/StatefulPartitionedCall"^dense_843/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 2F
!dense_840/StatefulPartitionedCall!dense_840/StatefulPartitionedCall2F
!dense_841/StatefulPartitionedCall!dense_841/StatefulPartitionedCall2F
!dense_842/StatefulPartitionedCall!dense_842/StatefulPartitionedCall2F
!dense_843/StatefulPartitionedCall!dense_843/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
G__inference_dense_843_layer_call_and_return_conditional_losses_18624181

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
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
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
,__inference_dense_841_layer_call_fn_18624130

inputs
unknown:

	unknown_0:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_841_layer_call_and_return_conditional_losses_18623731o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
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
?

?
G__inference_dense_840_layer_call_and_return_conditional_losses_18624121

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
,__inference_dense_842_layer_call_fn_18624150

inputs
unknown:

	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_842_layer_call_and_return_conditional_losses_18623748o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????
: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?+
?
#__inference__wrapped_model_18623696
	input_211D
2model_210_dense_840_matmul_readvariableop_resource:A
3model_210_dense_840_biasadd_readvariableop_resource:D
2model_210_dense_841_matmul_readvariableop_resource:
A
3model_210_dense_841_biasadd_readvariableop_resource:
D
2model_210_dense_842_matmul_readvariableop_resource:
A
3model_210_dense_842_biasadd_readvariableop_resource:D
2model_210_dense_843_matmul_readvariableop_resource:A
3model_210_dense_843_biasadd_readvariableop_resource:
identity??*model_210/dense_840/BiasAdd/ReadVariableOp?)model_210/dense_840/MatMul/ReadVariableOp?*model_210/dense_841/BiasAdd/ReadVariableOp?)model_210/dense_841/MatMul/ReadVariableOp?*model_210/dense_842/BiasAdd/ReadVariableOp?)model_210/dense_842/MatMul/ReadVariableOp?*model_210/dense_843/BiasAdd/ReadVariableOp?)model_210/dense_843/MatMul/ReadVariableOp?
)model_210/dense_840/MatMul/ReadVariableOpReadVariableOp2model_210_dense_840_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
model_210/dense_840/MatMulMatMul	input_2111model_210/dense_840/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
*model_210/dense_840/BiasAdd/ReadVariableOpReadVariableOp3model_210_dense_840_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model_210/dense_840/BiasAddBiasAdd$model_210/dense_840/MatMul:product:02model_210/dense_840/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????x
model_210/dense_840/ReluRelu$model_210/dense_840/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
)model_210/dense_841/MatMul/ReadVariableOpReadVariableOp2model_210_dense_841_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0?
model_210/dense_841/MatMulMatMul&model_210/dense_840/Relu:activations:01model_210/dense_841/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
*model_210/dense_841/BiasAdd/ReadVariableOpReadVariableOp3model_210_dense_841_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0?
model_210/dense_841/BiasAddBiasAdd$model_210/dense_841/MatMul:product:02model_210/dense_841/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
x
model_210/dense_841/ReluRelu$model_210/dense_841/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
?
)model_210/dense_842/MatMul/ReadVariableOpReadVariableOp2model_210_dense_842_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0?
model_210/dense_842/MatMulMatMul&model_210/dense_841/Relu:activations:01model_210/dense_842/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
*model_210/dense_842/BiasAdd/ReadVariableOpReadVariableOp3model_210_dense_842_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model_210/dense_842/BiasAddBiasAdd$model_210/dense_842/MatMul:product:02model_210/dense_842/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????x
model_210/dense_842/ReluRelu$model_210/dense_842/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
)model_210/dense_843/MatMul/ReadVariableOpReadVariableOp2model_210_dense_843_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
model_210/dense_843/MatMulMatMul&model_210/dense_842/Relu:activations:01model_210/dense_843/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
*model_210/dense_843/BiasAdd/ReadVariableOpReadVariableOp3model_210_dense_843_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model_210/dense_843/BiasAddBiasAdd$model_210/dense_843/MatMul:product:02model_210/dense_843/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????~
model_210/dense_843/SigmoidSigmoid$model_210/dense_843/BiasAdd:output:0*
T0*'
_output_shapes
:?????????n
IdentityIdentitymodel_210/dense_843/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp+^model_210/dense_840/BiasAdd/ReadVariableOp*^model_210/dense_840/MatMul/ReadVariableOp+^model_210/dense_841/BiasAdd/ReadVariableOp*^model_210/dense_841/MatMul/ReadVariableOp+^model_210/dense_842/BiasAdd/ReadVariableOp*^model_210/dense_842/MatMul/ReadVariableOp+^model_210/dense_843/BiasAdd/ReadVariableOp*^model_210/dense_843/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 2X
*model_210/dense_840/BiasAdd/ReadVariableOp*model_210/dense_840/BiasAdd/ReadVariableOp2V
)model_210/dense_840/MatMul/ReadVariableOp)model_210/dense_840/MatMul/ReadVariableOp2X
*model_210/dense_841/BiasAdd/ReadVariableOp*model_210/dense_841/BiasAdd/ReadVariableOp2V
)model_210/dense_841/MatMul/ReadVariableOp)model_210/dense_841/MatMul/ReadVariableOp2X
*model_210/dense_842/BiasAdd/ReadVariableOp*model_210/dense_842/BiasAdd/ReadVariableOp2V
)model_210/dense_842/MatMul/ReadVariableOp)model_210/dense_842/MatMul/ReadVariableOp2X
*model_210/dense_843/BiasAdd/ReadVariableOp*model_210/dense_843/BiasAdd/ReadVariableOp2V
)model_210/dense_843/MatMul/ReadVariableOp)model_210/dense_843/MatMul/ReadVariableOp:R N
'
_output_shapes
:?????????
#
_user_specified_name	input_211
?%
?
G__inference_model_210_layer_call_and_return_conditional_losses_18624101

inputs:
(dense_840_matmul_readvariableop_resource:7
)dense_840_biasadd_readvariableop_resource::
(dense_841_matmul_readvariableop_resource:
7
)dense_841_biasadd_readvariableop_resource:
:
(dense_842_matmul_readvariableop_resource:
7
)dense_842_biasadd_readvariableop_resource::
(dense_843_matmul_readvariableop_resource:7
)dense_843_biasadd_readvariableop_resource:
identity?? dense_840/BiasAdd/ReadVariableOp?dense_840/MatMul/ReadVariableOp? dense_841/BiasAdd/ReadVariableOp?dense_841/MatMul/ReadVariableOp? dense_842/BiasAdd/ReadVariableOp?dense_842/MatMul/ReadVariableOp? dense_843/BiasAdd/ReadVariableOp?dense_843/MatMul/ReadVariableOp?
dense_840/MatMul/ReadVariableOpReadVariableOp(dense_840_matmul_readvariableop_resource*
_output_shapes

:*
dtype0}
dense_840/MatMulMatMulinputs'dense_840/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
 dense_840/BiasAdd/ReadVariableOpReadVariableOp)dense_840_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_840/BiasAddBiasAdddense_840/MatMul:product:0(dense_840/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d
dense_840/ReluReludense_840/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
dense_841/MatMul/ReadVariableOpReadVariableOp(dense_841_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0?
dense_841/MatMulMatMuldense_840/Relu:activations:0'dense_841/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
 dense_841/BiasAdd/ReadVariableOpReadVariableOp)dense_841_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0?
dense_841/BiasAddBiasAdddense_841/MatMul:product:0(dense_841/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
d
dense_841/ReluReludense_841/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
?
dense_842/MatMul/ReadVariableOpReadVariableOp(dense_842_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0?
dense_842/MatMulMatMuldense_841/Relu:activations:0'dense_842/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
 dense_842/BiasAdd/ReadVariableOpReadVariableOp)dense_842_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_842/BiasAddBiasAdddense_842/MatMul:product:0(dense_842/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d
dense_842/ReluReludense_842/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
dense_843/MatMul/ReadVariableOpReadVariableOp(dense_843_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
dense_843/MatMulMatMuldense_842/Relu:activations:0'dense_843/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
 dense_843/BiasAdd/ReadVariableOpReadVariableOp)dense_843_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_843/BiasAddBiasAdddense_843/MatMul:product:0(dense_843/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????j
dense_843/SigmoidSigmoiddense_843/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d
IdentityIdentitydense_843/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^dense_840/BiasAdd/ReadVariableOp ^dense_840/MatMul/ReadVariableOp!^dense_841/BiasAdd/ReadVariableOp ^dense_841/MatMul/ReadVariableOp!^dense_842/BiasAdd/ReadVariableOp ^dense_842/MatMul/ReadVariableOp!^dense_843/BiasAdd/ReadVariableOp ^dense_843/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 2D
 dense_840/BiasAdd/ReadVariableOp dense_840/BiasAdd/ReadVariableOp2B
dense_840/MatMul/ReadVariableOpdense_840/MatMul/ReadVariableOp2D
 dense_841/BiasAdd/ReadVariableOp dense_841/BiasAdd/ReadVariableOp2B
dense_841/MatMul/ReadVariableOpdense_841/MatMul/ReadVariableOp2D
 dense_842/BiasAdd/ReadVariableOp dense_842/BiasAdd/ReadVariableOp2B
dense_842/MatMul/ReadVariableOpdense_842/MatMul/ReadVariableOp2D
 dense_843/BiasAdd/ReadVariableOp dense_843/BiasAdd/ReadVariableOp2B
dense_843/MatMul/ReadVariableOpdense_843/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
G__inference_model_210_layer_call_and_return_conditional_losses_18623966
	input_211$
dense_840_18623945: 
dense_840_18623947:$
dense_841_18623950:
 
dense_841_18623952:
$
dense_842_18623955:
 
dense_842_18623957:$
dense_843_18623960: 
dense_843_18623962:
identity??!dense_840/StatefulPartitionedCall?!dense_841/StatefulPartitionedCall?!dense_842/StatefulPartitionedCall?!dense_843/StatefulPartitionedCall?
!dense_840/StatefulPartitionedCallStatefulPartitionedCall	input_211dense_840_18623945dense_840_18623947*
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
G__inference_dense_840_layer_call_and_return_conditional_losses_18623714?
!dense_841/StatefulPartitionedCallStatefulPartitionedCall*dense_840/StatefulPartitionedCall:output:0dense_841_18623950dense_841_18623952*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_841_layer_call_and_return_conditional_losses_18623731?
!dense_842/StatefulPartitionedCallStatefulPartitionedCall*dense_841/StatefulPartitionedCall:output:0dense_842_18623955dense_842_18623957*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_842_layer_call_and_return_conditional_losses_18623748?
!dense_843/StatefulPartitionedCallStatefulPartitionedCall*dense_842/StatefulPartitionedCall:output:0dense_843_18623960dense_843_18623962*
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
G__inference_dense_843_layer_call_and_return_conditional_losses_18623765y
IdentityIdentity*dense_843/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp"^dense_840/StatefulPartitionedCall"^dense_841/StatefulPartitionedCall"^dense_842/StatefulPartitionedCall"^dense_843/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 2F
!dense_840/StatefulPartitionedCall!dense_840/StatefulPartitionedCall2F
!dense_841/StatefulPartitionedCall!dense_841/StatefulPartitionedCall2F
!dense_842/StatefulPartitionedCall!dense_842/StatefulPartitionedCall2F
!dense_843/StatefulPartitionedCall!dense_843/StatefulPartitionedCall:R N
'
_output_shapes
:?????????
#
_user_specified_name	input_211
?

?
G__inference_dense_841_layer_call_and_return_conditional_losses_18624141

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????
a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????
w
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
?C
?
!__inference__traced_save_18624297
file_prefix/
+savev2_dense_840_kernel_read_readvariableop-
)savev2_dense_840_bias_read_readvariableop/
+savev2_dense_841_kernel_read_readvariableop-
)savev2_dense_841_bias_read_readvariableop/
+savev2_dense_842_kernel_read_readvariableop-
)savev2_dense_842_bias_read_readvariableop/
+savev2_dense_843_kernel_read_readvariableop-
)savev2_dense_843_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop6
2savev2_adam_dense_840_kernel_m_read_readvariableop4
0savev2_adam_dense_840_bias_m_read_readvariableop6
2savev2_adam_dense_841_kernel_m_read_readvariableop4
0savev2_adam_dense_841_bias_m_read_readvariableop6
2savev2_adam_dense_842_kernel_m_read_readvariableop4
0savev2_adam_dense_842_bias_m_read_readvariableop6
2savev2_adam_dense_843_kernel_m_read_readvariableop4
0savev2_adam_dense_843_bias_m_read_readvariableop6
2savev2_adam_dense_840_kernel_v_read_readvariableop4
0savev2_adam_dense_840_bias_v_read_readvariableop6
2savev2_adam_dense_841_kernel_v_read_readvariableop4
0savev2_adam_dense_841_bias_v_read_readvariableop6
2savev2_adam_dense_842_kernel_v_read_readvariableop4
0savev2_adam_dense_842_bias_v_read_readvariableop6
2savev2_adam_dense_843_kernel_v_read_readvariableop4
0savev2_adam_dense_843_bias_v_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_dense_840_kernel_read_readvariableop)savev2_dense_840_bias_read_readvariableop+savev2_dense_841_kernel_read_readvariableop)savev2_dense_841_bias_read_readvariableop+savev2_dense_842_kernel_read_readvariableop)savev2_dense_842_bias_read_readvariableop+savev2_dense_843_kernel_read_readvariableop)savev2_dense_843_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop2savev2_adam_dense_840_kernel_m_read_readvariableop0savev2_adam_dense_840_bias_m_read_readvariableop2savev2_adam_dense_841_kernel_m_read_readvariableop0savev2_adam_dense_841_bias_m_read_readvariableop2savev2_adam_dense_842_kernel_m_read_readvariableop0savev2_adam_dense_842_bias_m_read_readvariableop2savev2_adam_dense_843_kernel_m_read_readvariableop0savev2_adam_dense_843_bias_m_read_readvariableop2savev2_adam_dense_840_kernel_v_read_readvariableop0savev2_adam_dense_840_bias_v_read_readvariableop2savev2_adam_dense_841_kernel_v_read_readvariableop0savev2_adam_dense_841_bias_v_read_readvariableop2savev2_adam_dense_842_kernel_v_read_readvariableop0savev2_adam_dense_842_bias_v_read_readvariableop2savev2_adam_dense_843_kernel_v_read_readvariableop0savev2_adam_dense_843_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
?: :::
:
:
:::: : : : : : : :::
:
:
::::::
:
:
:::: 2(
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

:
: 

_output_shapes
:
:$ 

_output_shapes

:
: 

_output_shapes
::$ 

_output_shapes

:: 
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

:
: 

_output_shapes
:
:$ 

_output_shapes

:
: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:
: 

_output_shapes
:
:$ 

_output_shapes

:
: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
:: 

_output_shapes
: 
?	
?
,__inference_model_210_layer_call_fn_18624037

inputs
unknown:
	unknown_0:
	unknown_1:

	unknown_2:

	unknown_3:

	unknown_4:
	unknown_5:
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
G__inference_model_210_layer_call_and_return_conditional_losses_18623878o
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
,__inference_dense_840_layer_call_fn_18624110

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
G__inference_dense_840_layer_call_and_return_conditional_losses_18623714o
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
?%
?
G__inference_model_210_layer_call_and_return_conditional_losses_18624069

inputs:
(dense_840_matmul_readvariableop_resource:7
)dense_840_biasadd_readvariableop_resource::
(dense_841_matmul_readvariableop_resource:
7
)dense_841_biasadd_readvariableop_resource:
:
(dense_842_matmul_readvariableop_resource:
7
)dense_842_biasadd_readvariableop_resource::
(dense_843_matmul_readvariableop_resource:7
)dense_843_biasadd_readvariableop_resource:
identity?? dense_840/BiasAdd/ReadVariableOp?dense_840/MatMul/ReadVariableOp? dense_841/BiasAdd/ReadVariableOp?dense_841/MatMul/ReadVariableOp? dense_842/BiasAdd/ReadVariableOp?dense_842/MatMul/ReadVariableOp? dense_843/BiasAdd/ReadVariableOp?dense_843/MatMul/ReadVariableOp?
dense_840/MatMul/ReadVariableOpReadVariableOp(dense_840_matmul_readvariableop_resource*
_output_shapes

:*
dtype0}
dense_840/MatMulMatMulinputs'dense_840/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
 dense_840/BiasAdd/ReadVariableOpReadVariableOp)dense_840_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_840/BiasAddBiasAdddense_840/MatMul:product:0(dense_840/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d
dense_840/ReluReludense_840/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
dense_841/MatMul/ReadVariableOpReadVariableOp(dense_841_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0?
dense_841/MatMulMatMuldense_840/Relu:activations:0'dense_841/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
 dense_841/BiasAdd/ReadVariableOpReadVariableOp)dense_841_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0?
dense_841/BiasAddBiasAdddense_841/MatMul:product:0(dense_841/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
d
dense_841/ReluReludense_841/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
?
dense_842/MatMul/ReadVariableOpReadVariableOp(dense_842_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0?
dense_842/MatMulMatMuldense_841/Relu:activations:0'dense_842/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
 dense_842/BiasAdd/ReadVariableOpReadVariableOp)dense_842_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_842/BiasAddBiasAdddense_842/MatMul:product:0(dense_842/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d
dense_842/ReluReludense_842/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
dense_843/MatMul/ReadVariableOpReadVariableOp(dense_843_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
dense_843/MatMulMatMuldense_842/Relu:activations:0'dense_843/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
 dense_843/BiasAdd/ReadVariableOpReadVariableOp)dense_843_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_843/BiasAddBiasAdddense_843/MatMul:product:0(dense_843/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????j
dense_843/SigmoidSigmoiddense_843/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d
IdentityIdentitydense_843/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^dense_840/BiasAdd/ReadVariableOp ^dense_840/MatMul/ReadVariableOp!^dense_841/BiasAdd/ReadVariableOp ^dense_841/MatMul/ReadVariableOp!^dense_842/BiasAdd/ReadVariableOp ^dense_842/MatMul/ReadVariableOp!^dense_843/BiasAdd/ReadVariableOp ^dense_843/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 2D
 dense_840/BiasAdd/ReadVariableOp dense_840/BiasAdd/ReadVariableOp2B
dense_840/MatMul/ReadVariableOpdense_840/MatMul/ReadVariableOp2D
 dense_841/BiasAdd/ReadVariableOp dense_841/BiasAdd/ReadVariableOp2B
dense_841/MatMul/ReadVariableOpdense_841/MatMul/ReadVariableOp2D
 dense_842/BiasAdd/ReadVariableOp dense_842/BiasAdd/ReadVariableOp2B
dense_842/MatMul/ReadVariableOpdense_842/MatMul/ReadVariableOp2D
 dense_843/BiasAdd/ReadVariableOp dense_843/BiasAdd/ReadVariableOp2B
dense_843/MatMul/ReadVariableOpdense_843/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
G__inference_model_210_layer_call_and_return_conditional_losses_18623772

inputs$
dense_840_18623715: 
dense_840_18623717:$
dense_841_18623732:
 
dense_841_18623734:
$
dense_842_18623749:
 
dense_842_18623751:$
dense_843_18623766: 
dense_843_18623768:
identity??!dense_840/StatefulPartitionedCall?!dense_841/StatefulPartitionedCall?!dense_842/StatefulPartitionedCall?!dense_843/StatefulPartitionedCall?
!dense_840/StatefulPartitionedCallStatefulPartitionedCallinputsdense_840_18623715dense_840_18623717*
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
G__inference_dense_840_layer_call_and_return_conditional_losses_18623714?
!dense_841/StatefulPartitionedCallStatefulPartitionedCall*dense_840/StatefulPartitionedCall:output:0dense_841_18623732dense_841_18623734*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_841_layer_call_and_return_conditional_losses_18623731?
!dense_842/StatefulPartitionedCallStatefulPartitionedCall*dense_841/StatefulPartitionedCall:output:0dense_842_18623749dense_842_18623751*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_842_layer_call_and_return_conditional_losses_18623748?
!dense_843/StatefulPartitionedCallStatefulPartitionedCall*dense_842/StatefulPartitionedCall:output:0dense_843_18623766dense_843_18623768*
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
G__inference_dense_843_layer_call_and_return_conditional_losses_18623765y
IdentityIdentity*dense_843/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp"^dense_840/StatefulPartitionedCall"^dense_841/StatefulPartitionedCall"^dense_842/StatefulPartitionedCall"^dense_843/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 2F
!dense_840/StatefulPartitionedCall!dense_840/StatefulPartitionedCall2F
!dense_841/StatefulPartitionedCall!dense_841/StatefulPartitionedCall2F
!dense_842/StatefulPartitionedCall!dense_842/StatefulPartitionedCall2F
!dense_843/StatefulPartitionedCall!dense_843/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
G__inference_model_210_layer_call_and_return_conditional_losses_18623942
	input_211$
dense_840_18623921: 
dense_840_18623923:$
dense_841_18623926:
 
dense_841_18623928:
$
dense_842_18623931:
 
dense_842_18623933:$
dense_843_18623936: 
dense_843_18623938:
identity??!dense_840/StatefulPartitionedCall?!dense_841/StatefulPartitionedCall?!dense_842/StatefulPartitionedCall?!dense_843/StatefulPartitionedCall?
!dense_840/StatefulPartitionedCallStatefulPartitionedCall	input_211dense_840_18623921dense_840_18623923*
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
G__inference_dense_840_layer_call_and_return_conditional_losses_18623714?
!dense_841/StatefulPartitionedCallStatefulPartitionedCall*dense_840/StatefulPartitionedCall:output:0dense_841_18623926dense_841_18623928*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_841_layer_call_and_return_conditional_losses_18623731?
!dense_842/StatefulPartitionedCallStatefulPartitionedCall*dense_841/StatefulPartitionedCall:output:0dense_842_18623931dense_842_18623933*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_842_layer_call_and_return_conditional_losses_18623748?
!dense_843/StatefulPartitionedCallStatefulPartitionedCall*dense_842/StatefulPartitionedCall:output:0dense_843_18623936dense_843_18623938*
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
G__inference_dense_843_layer_call_and_return_conditional_losses_18623765y
IdentityIdentity*dense_843/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp"^dense_840/StatefulPartitionedCall"^dense_841/StatefulPartitionedCall"^dense_842/StatefulPartitionedCall"^dense_843/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 2F
!dense_840/StatefulPartitionedCall!dense_840/StatefulPartitionedCall2F
!dense_841/StatefulPartitionedCall!dense_841/StatefulPartitionedCall2F
!dense_842/StatefulPartitionedCall!dense_842/StatefulPartitionedCall2F
!dense_843/StatefulPartitionedCall!dense_843/StatefulPartitionedCall:R N
'
_output_shapes
:?????????
#
_user_specified_name	input_211
?	
?
,__inference_model_210_layer_call_fn_18623918
	input_211
unknown:
	unknown_0:
	unknown_1:

	unknown_2:

	unknown_3:

	unknown_4:
	unknown_5:
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall	input_211unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
G__inference_model_210_layer_call_and_return_conditional_losses_18623878o
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
_user_specified_name	input_211
?
?
,__inference_dense_843_layer_call_fn_18624170

inputs
unknown:
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
G__inference_dense_843_layer_call_and_return_conditional_losses_18623765o
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
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
,__inference_model_210_layer_call_fn_18624016

inputs
unknown:
	unknown_0:
	unknown_1:

	unknown_2:

	unknown_3:

	unknown_4:
	unknown_5:
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
G__inference_model_210_layer_call_and_return_conditional_losses_18623772o
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
?}
?
$__inference__traced_restore_18624400
file_prefix3
!assignvariableop_dense_840_kernel:/
!assignvariableop_1_dense_840_bias:5
#assignvariableop_2_dense_841_kernel:
/
!assignvariableop_3_dense_841_bias:
5
#assignvariableop_4_dense_842_kernel:
/
!assignvariableop_5_dense_842_bias:5
#assignvariableop_6_dense_843_kernel:/
!assignvariableop_7_dense_843_bias:&
assignvariableop_8_adam_iter:	 (
assignvariableop_9_adam_beta_1: )
assignvariableop_10_adam_beta_2: (
assignvariableop_11_adam_decay: 0
&assignvariableop_12_adam_learning_rate: #
assignvariableop_13_total: #
assignvariableop_14_count: =
+assignvariableop_15_adam_dense_840_kernel_m:7
)assignvariableop_16_adam_dense_840_bias_m:=
+assignvariableop_17_adam_dense_841_kernel_m:
7
)assignvariableop_18_adam_dense_841_bias_m:
=
+assignvariableop_19_adam_dense_842_kernel_m:
7
)assignvariableop_20_adam_dense_842_bias_m:=
+assignvariableop_21_adam_dense_843_kernel_m:7
)assignvariableop_22_adam_dense_843_bias_m:=
+assignvariableop_23_adam_dense_840_kernel_v:7
)assignvariableop_24_adam_dense_840_bias_v:=
+assignvariableop_25_adam_dense_841_kernel_v:
7
)assignvariableop_26_adam_dense_841_bias_v:
=
+assignvariableop_27_adam_dense_842_kernel_v:
7
)assignvariableop_28_adam_dense_842_bias_v:=
+assignvariableop_29_adam_dense_843_kernel_v:7
)assignvariableop_30_adam_dense_843_bias_v:
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
AssignVariableOpAssignVariableOp!assignvariableop_dense_840_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp!assignvariableop_1_dense_840_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp#assignvariableop_2_dense_841_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp!assignvariableop_3_dense_841_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp#assignvariableop_4_dense_842_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp!assignvariableop_5_dense_842_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp#assignvariableop_6_dense_843_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp!assignvariableop_7_dense_843_biasIdentity_7:output:0"/device:CPU:0*
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
AssignVariableOp_15AssignVariableOp+assignvariableop_15_adam_dense_840_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp)assignvariableop_16_adam_dense_840_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp+assignvariableop_17_adam_dense_841_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp)assignvariableop_18_adam_dense_841_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp+assignvariableop_19_adam_dense_842_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp)assignvariableop_20_adam_dense_842_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp+assignvariableop_21_adam_dense_843_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp)assignvariableop_22_adam_dense_843_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp+assignvariableop_23_adam_dense_840_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp)assignvariableop_24_adam_dense_840_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp+assignvariableop_25_adam_dense_841_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOp)assignvariableop_26_adam_dense_841_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOp+assignvariableop_27_adam_dense_842_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOp)assignvariableop_28_adam_dense_842_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOp+assignvariableop_29_adam_dense_843_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOp)assignvariableop_30_adam_dense_843_bias_vIdentity_30:output:0"/device:CPU:0*
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
G__inference_dense_840_layer_call_and_return_conditional_losses_18623714

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
?	
?
&__inference_signature_wrapper_18623995
	input_211
unknown:
	unknown_0:
	unknown_1:

	unknown_2:

	unknown_3:

	unknown_4:
	unknown_5:
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall	input_211unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
#__inference__wrapped_model_18623696o
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
_user_specified_name	input_211
?	
?
,__inference_model_210_layer_call_fn_18623791
	input_211
unknown:
	unknown_0:
	unknown_1:

	unknown_2:

	unknown_3:

	unknown_4:
	unknown_5:
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall	input_211unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
G__inference_model_210_layer_call_and_return_conditional_losses_18623772o
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
_user_specified_name	input_211
?

?
G__inference_dense_842_layer_call_and_return_conditional_losses_18623748

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?

?
G__inference_dense_843_layer_call_and_return_conditional_losses_18623765

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
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
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
G__inference_dense_842_layer_call_and_return_conditional_losses_18624161

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????
: : 20
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
	input_2112
serving_default_input_211:0?????????=
	dense_8430
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
": 2dense_840/kernel
:2dense_840/bias
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
": 
2dense_841/kernel
:
2dense_841/bias
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
": 
2dense_842/kernel
:2dense_842/bias
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
": 2dense_843/kernel
:2dense_843/bias
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
':%2Adam/dense_840/kernel/m
!:2Adam/dense_840/bias/m
':%
2Adam/dense_841/kernel/m
!:
2Adam/dense_841/bias/m
':%
2Adam/dense_842/kernel/m
!:2Adam/dense_842/bias/m
':%2Adam/dense_843/kernel/m
!:2Adam/dense_843/bias/m
':%2Adam/dense_840/kernel/v
!:2Adam/dense_840/bias/v
':%
2Adam/dense_841/kernel/v
!:
2Adam/dense_841/bias/v
':%
2Adam/dense_842/kernel/v
!:2Adam/dense_842/bias/v
':%2Adam/dense_843/kernel/v
!:2Adam/dense_843/bias/v
?2?
,__inference_model_210_layer_call_fn_18623791
,__inference_model_210_layer_call_fn_18624016
,__inference_model_210_layer_call_fn_18624037
,__inference_model_210_layer_call_fn_18623918?
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
G__inference_model_210_layer_call_and_return_conditional_losses_18624069
G__inference_model_210_layer_call_and_return_conditional_losses_18624101
G__inference_model_210_layer_call_and_return_conditional_losses_18623942
G__inference_model_210_layer_call_and_return_conditional_losses_18623966?
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
#__inference__wrapped_model_18623696	input_211"?
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
,__inference_dense_840_layer_call_fn_18624110?
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
G__inference_dense_840_layer_call_and_return_conditional_losses_18624121?
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
,__inference_dense_841_layer_call_fn_18624130?
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
G__inference_dense_841_layer_call_and_return_conditional_losses_18624141?
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
,__inference_dense_842_layer_call_fn_18624150?
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
G__inference_dense_842_layer_call_and_return_conditional_losses_18624161?
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
,__inference_dense_843_layer_call_fn_18624170?
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
G__inference_dense_843_layer_call_and_return_conditional_losses_18624181?
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
&__inference_signature_wrapper_18623995	input_211"?
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
#__inference__wrapped_model_18623696u2?/
(?%
#? 
	input_211?????????
? "5?2
0
	dense_843#? 
	dense_843??????????
G__inference_dense_840_layer_call_and_return_conditional_losses_18624121\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? 
,__inference_dense_840_layer_call_fn_18624110O/?,
%?"
 ?
inputs?????????
? "???????????
G__inference_dense_841_layer_call_and_return_conditional_losses_18624141\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????

? 
,__inference_dense_841_layer_call_fn_18624130O/?,
%?"
 ?
inputs?????????
? "??????????
?
G__inference_dense_842_layer_call_and_return_conditional_losses_18624161\/?,
%?"
 ?
inputs?????????

? "%?"
?
0?????????
? 
,__inference_dense_842_layer_call_fn_18624150O/?,
%?"
 ?
inputs?????????

? "???????????
G__inference_dense_843_layer_call_and_return_conditional_losses_18624181\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? 
,__inference_dense_843_layer_call_fn_18624170O/?,
%?"
 ?
inputs?????????
? "???????????
G__inference_model_210_layer_call_and_return_conditional_losses_18623942m:?7
0?-
#? 
	input_211?????????
p 

 
? "%?"
?
0?????????
? ?
G__inference_model_210_layer_call_and_return_conditional_losses_18623966m:?7
0?-
#? 
	input_211?????????
p

 
? "%?"
?
0?????????
? ?
G__inference_model_210_layer_call_and_return_conditional_losses_18624069j7?4
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
G__inference_model_210_layer_call_and_return_conditional_losses_18624101j7?4
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
,__inference_model_210_layer_call_fn_18623791`:?7
0?-
#? 
	input_211?????????
p 

 
? "???????????
,__inference_model_210_layer_call_fn_18623918`:?7
0?-
#? 
	input_211?????????
p

 
? "???????????
,__inference_model_210_layer_call_fn_18624016]7?4
-?*
 ?
inputs?????????
p 

 
? "???????????
,__inference_model_210_layer_call_fn_18624037]7?4
-?*
 ?
inputs?????????
p

 
? "???????????
&__inference_signature_wrapper_18623995???<
? 
5?2
0
	input_211#? 
	input_211?????????"5?2
0
	dense_843#? 
	dense_843?????????
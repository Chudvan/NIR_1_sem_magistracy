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
dense_784/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_784/kernel
u
$dense_784/kernel/Read/ReadVariableOpReadVariableOpdense_784/kernel*
_output_shapes

:*
dtype0
t
dense_784/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_784/bias
m
"dense_784/bias/Read/ReadVariableOpReadVariableOpdense_784/bias*
_output_shapes
:*
dtype0
|
dense_785/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_785/kernel
u
$dense_785/kernel/Read/ReadVariableOpReadVariableOpdense_785/kernel*
_output_shapes

:*
dtype0
t
dense_785/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_785/bias
m
"dense_785/bias/Read/ReadVariableOpReadVariableOpdense_785/bias*
_output_shapes
:*
dtype0
|
dense_786/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_786/kernel
u
$dense_786/kernel/Read/ReadVariableOpReadVariableOpdense_786/kernel*
_output_shapes

:*
dtype0
t
dense_786/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_786/bias
m
"dense_786/bias/Read/ReadVariableOpReadVariableOpdense_786/bias*
_output_shapes
:*
dtype0
|
dense_787/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_787/kernel
u
$dense_787/kernel/Read/ReadVariableOpReadVariableOpdense_787/kernel*
_output_shapes

:*
dtype0
t
dense_787/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_787/bias
m
"dense_787/bias/Read/ReadVariableOpReadVariableOpdense_787/bias*
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
Adam/dense_784/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_784/kernel/m
?
+Adam/dense_784/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_784/kernel/m*
_output_shapes

:*
dtype0
?
Adam/dense_784/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_784/bias/m
{
)Adam/dense_784/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_784/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_785/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_785/kernel/m
?
+Adam/dense_785/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_785/kernel/m*
_output_shapes

:*
dtype0
?
Adam/dense_785/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_785/bias/m
{
)Adam/dense_785/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_785/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_786/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_786/kernel/m
?
+Adam/dense_786/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_786/kernel/m*
_output_shapes

:*
dtype0
?
Adam/dense_786/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_786/bias/m
{
)Adam/dense_786/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_786/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_787/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_787/kernel/m
?
+Adam/dense_787/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_787/kernel/m*
_output_shapes

:*
dtype0
?
Adam/dense_787/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_787/bias/m
{
)Adam/dense_787/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_787/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_784/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_784/kernel/v
?
+Adam/dense_784/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_784/kernel/v*
_output_shapes

:*
dtype0
?
Adam/dense_784/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_784/bias/v
{
)Adam/dense_784/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_784/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense_785/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_785/kernel/v
?
+Adam/dense_785/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_785/kernel/v*
_output_shapes

:*
dtype0
?
Adam/dense_785/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_785/bias/v
{
)Adam/dense_785/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_785/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense_786/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_786/kernel/v
?
+Adam/dense_786/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_786/kernel/v*
_output_shapes

:*
dtype0
?
Adam/dense_786/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_786/bias/v
{
)Adam/dense_786/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_786/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense_787/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_787/kernel/v
?
+Adam/dense_787/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_787/kernel/v*
_output_shapes

:*
dtype0
?
Adam/dense_787/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_787/bias/v
{
)Adam/dense_787/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_787/bias/v*
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
VARIABLE_VALUEdense_784/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_784/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEdense_785/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_785/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEdense_786/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_786/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEdense_787/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_787/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEAdam/dense_784/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_784/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_785/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_785/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_786/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_786/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_787/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_787/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_784/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_784/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_785/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_785/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_786/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_786/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_787/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_787/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|
serving_default_input_197Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_197dense_784/kerneldense_784/biasdense_785/kerneldense_785/biasdense_786/kerneldense_786/biasdense_787/kerneldense_787/bias*
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
&__inference_signature_wrapper_18611647
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$dense_784/kernel/Read/ReadVariableOp"dense_784/bias/Read/ReadVariableOp$dense_785/kernel/Read/ReadVariableOp"dense_785/bias/Read/ReadVariableOp$dense_786/kernel/Read/ReadVariableOp"dense_786/bias/Read/ReadVariableOp$dense_787/kernel/Read/ReadVariableOp"dense_787/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp+Adam/dense_784/kernel/m/Read/ReadVariableOp)Adam/dense_784/bias/m/Read/ReadVariableOp+Adam/dense_785/kernel/m/Read/ReadVariableOp)Adam/dense_785/bias/m/Read/ReadVariableOp+Adam/dense_786/kernel/m/Read/ReadVariableOp)Adam/dense_786/bias/m/Read/ReadVariableOp+Adam/dense_787/kernel/m/Read/ReadVariableOp)Adam/dense_787/bias/m/Read/ReadVariableOp+Adam/dense_784/kernel/v/Read/ReadVariableOp)Adam/dense_784/bias/v/Read/ReadVariableOp+Adam/dense_785/kernel/v/Read/ReadVariableOp)Adam/dense_785/bias/v/Read/ReadVariableOp+Adam/dense_786/kernel/v/Read/ReadVariableOp)Adam/dense_786/bias/v/Read/ReadVariableOp+Adam/dense_787/kernel/v/Read/ReadVariableOp)Adam/dense_787/bias/v/Read/ReadVariableOpConst*,
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
!__inference__traced_save_18611949
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_784/kerneldense_784/biasdense_785/kerneldense_785/biasdense_786/kerneldense_786/biasdense_787/kerneldense_787/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/dense_784/kernel/mAdam/dense_784/bias/mAdam/dense_785/kernel/mAdam/dense_785/bias/mAdam/dense_786/kernel/mAdam/dense_786/bias/mAdam/dense_787/kernel/mAdam/dense_787/bias/mAdam/dense_784/kernel/vAdam/dense_784/bias/vAdam/dense_785/kernel/vAdam/dense_785/bias/vAdam/dense_786/kernel/vAdam/dense_786/bias/vAdam/dense_787/kernel/vAdam/dense_787/bias/v*+
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
$__inference__traced_restore_18612052??
?
?
G__inference_model_196_layer_call_and_return_conditional_losses_18611530

inputs$
dense_784_18611509: 
dense_784_18611511:$
dense_785_18611514: 
dense_785_18611516:$
dense_786_18611519: 
dense_786_18611521:$
dense_787_18611524: 
dense_787_18611526:
identity??!dense_784/StatefulPartitionedCall?!dense_785/StatefulPartitionedCall?!dense_786/StatefulPartitionedCall?!dense_787/StatefulPartitionedCall?
!dense_784/StatefulPartitionedCallStatefulPartitionedCallinputsdense_784_18611509dense_784_18611511*
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
G__inference_dense_784_layer_call_and_return_conditional_losses_18611366?
!dense_785/StatefulPartitionedCallStatefulPartitionedCall*dense_784/StatefulPartitionedCall:output:0dense_785_18611514dense_785_18611516*
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
G__inference_dense_785_layer_call_and_return_conditional_losses_18611383?
!dense_786/StatefulPartitionedCallStatefulPartitionedCall*dense_785/StatefulPartitionedCall:output:0dense_786_18611519dense_786_18611521*
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
G__inference_dense_786_layer_call_and_return_conditional_losses_18611400?
!dense_787/StatefulPartitionedCallStatefulPartitionedCall*dense_786/StatefulPartitionedCall:output:0dense_787_18611524dense_787_18611526*
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
G__inference_dense_787_layer_call_and_return_conditional_losses_18611417y
IdentityIdentity*dense_787/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp"^dense_784/StatefulPartitionedCall"^dense_785/StatefulPartitionedCall"^dense_786/StatefulPartitionedCall"^dense_787/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 2F
!dense_784/StatefulPartitionedCall!dense_784/StatefulPartitionedCall2F
!dense_785/StatefulPartitionedCall!dense_785/StatefulPartitionedCall2F
!dense_786/StatefulPartitionedCall!dense_786/StatefulPartitionedCall2F
!dense_787/StatefulPartitionedCall!dense_787/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
G__inference_model_196_layer_call_and_return_conditional_losses_18611424

inputs$
dense_784_18611367: 
dense_784_18611369:$
dense_785_18611384: 
dense_785_18611386:$
dense_786_18611401: 
dense_786_18611403:$
dense_787_18611418: 
dense_787_18611420:
identity??!dense_784/StatefulPartitionedCall?!dense_785/StatefulPartitionedCall?!dense_786/StatefulPartitionedCall?!dense_787/StatefulPartitionedCall?
!dense_784/StatefulPartitionedCallStatefulPartitionedCallinputsdense_784_18611367dense_784_18611369*
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
G__inference_dense_784_layer_call_and_return_conditional_losses_18611366?
!dense_785/StatefulPartitionedCallStatefulPartitionedCall*dense_784/StatefulPartitionedCall:output:0dense_785_18611384dense_785_18611386*
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
G__inference_dense_785_layer_call_and_return_conditional_losses_18611383?
!dense_786/StatefulPartitionedCallStatefulPartitionedCall*dense_785/StatefulPartitionedCall:output:0dense_786_18611401dense_786_18611403*
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
G__inference_dense_786_layer_call_and_return_conditional_losses_18611400?
!dense_787/StatefulPartitionedCallStatefulPartitionedCall*dense_786/StatefulPartitionedCall:output:0dense_787_18611418dense_787_18611420*
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
G__inference_dense_787_layer_call_and_return_conditional_losses_18611417y
IdentityIdentity*dense_787/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp"^dense_784/StatefulPartitionedCall"^dense_785/StatefulPartitionedCall"^dense_786/StatefulPartitionedCall"^dense_787/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 2F
!dense_784/StatefulPartitionedCall!dense_784/StatefulPartitionedCall2F
!dense_785/StatefulPartitionedCall!dense_785/StatefulPartitionedCall2F
!dense_786/StatefulPartitionedCall!dense_786/StatefulPartitionedCall2F
!dense_787/StatefulPartitionedCall!dense_787/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?}
?
$__inference__traced_restore_18612052
file_prefix3
!assignvariableop_dense_784_kernel:/
!assignvariableop_1_dense_784_bias:5
#assignvariableop_2_dense_785_kernel:/
!assignvariableop_3_dense_785_bias:5
#assignvariableop_4_dense_786_kernel:/
!assignvariableop_5_dense_786_bias:5
#assignvariableop_6_dense_787_kernel:/
!assignvariableop_7_dense_787_bias:&
assignvariableop_8_adam_iter:	 (
assignvariableop_9_adam_beta_1: )
assignvariableop_10_adam_beta_2: (
assignvariableop_11_adam_decay: 0
&assignvariableop_12_adam_learning_rate: #
assignvariableop_13_total: #
assignvariableop_14_count: =
+assignvariableop_15_adam_dense_784_kernel_m:7
)assignvariableop_16_adam_dense_784_bias_m:=
+assignvariableop_17_adam_dense_785_kernel_m:7
)assignvariableop_18_adam_dense_785_bias_m:=
+assignvariableop_19_adam_dense_786_kernel_m:7
)assignvariableop_20_adam_dense_786_bias_m:=
+assignvariableop_21_adam_dense_787_kernel_m:7
)assignvariableop_22_adam_dense_787_bias_m:=
+assignvariableop_23_adam_dense_784_kernel_v:7
)assignvariableop_24_adam_dense_784_bias_v:=
+assignvariableop_25_adam_dense_785_kernel_v:7
)assignvariableop_26_adam_dense_785_bias_v:=
+assignvariableop_27_adam_dense_786_kernel_v:7
)assignvariableop_28_adam_dense_786_bias_v:=
+assignvariableop_29_adam_dense_787_kernel_v:7
)assignvariableop_30_adam_dense_787_bias_v:
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
AssignVariableOpAssignVariableOp!assignvariableop_dense_784_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp!assignvariableop_1_dense_784_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp#assignvariableop_2_dense_785_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp!assignvariableop_3_dense_785_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp#assignvariableop_4_dense_786_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp!assignvariableop_5_dense_786_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp#assignvariableop_6_dense_787_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp!assignvariableop_7_dense_787_biasIdentity_7:output:0"/device:CPU:0*
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
AssignVariableOp_15AssignVariableOp+assignvariableop_15_adam_dense_784_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp)assignvariableop_16_adam_dense_784_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp+assignvariableop_17_adam_dense_785_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp)assignvariableop_18_adam_dense_785_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp+assignvariableop_19_adam_dense_786_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp)assignvariableop_20_adam_dense_786_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp+assignvariableop_21_adam_dense_787_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp)assignvariableop_22_adam_dense_787_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp+assignvariableop_23_adam_dense_784_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp)assignvariableop_24_adam_dense_784_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp+assignvariableop_25_adam_dense_785_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOp)assignvariableop_26_adam_dense_785_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOp+assignvariableop_27_adam_dense_786_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOp)assignvariableop_28_adam_dense_786_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOp+assignvariableop_29_adam_dense_787_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOp)assignvariableop_30_adam_dense_787_bias_vIdentity_30:output:0"/device:CPU:0*
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
G__inference_dense_784_layer_call_and_return_conditional_losses_18611773

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

?
G__inference_dense_785_layer_call_and_return_conditional_losses_18611383

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
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
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
,__inference_dense_784_layer_call_fn_18611762

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
G__inference_dense_784_layer_call_and_return_conditional_losses_18611366o
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
?C
?
!__inference__traced_save_18611949
file_prefix/
+savev2_dense_784_kernel_read_readvariableop-
)savev2_dense_784_bias_read_readvariableop/
+savev2_dense_785_kernel_read_readvariableop-
)savev2_dense_785_bias_read_readvariableop/
+savev2_dense_786_kernel_read_readvariableop-
)savev2_dense_786_bias_read_readvariableop/
+savev2_dense_787_kernel_read_readvariableop-
)savev2_dense_787_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop6
2savev2_adam_dense_784_kernel_m_read_readvariableop4
0savev2_adam_dense_784_bias_m_read_readvariableop6
2savev2_adam_dense_785_kernel_m_read_readvariableop4
0savev2_adam_dense_785_bias_m_read_readvariableop6
2savev2_adam_dense_786_kernel_m_read_readvariableop4
0savev2_adam_dense_786_bias_m_read_readvariableop6
2savev2_adam_dense_787_kernel_m_read_readvariableop4
0savev2_adam_dense_787_bias_m_read_readvariableop6
2savev2_adam_dense_784_kernel_v_read_readvariableop4
0savev2_adam_dense_784_bias_v_read_readvariableop6
2savev2_adam_dense_785_kernel_v_read_readvariableop4
0savev2_adam_dense_785_bias_v_read_readvariableop6
2savev2_adam_dense_786_kernel_v_read_readvariableop4
0savev2_adam_dense_786_bias_v_read_readvariableop6
2savev2_adam_dense_787_kernel_v_read_readvariableop4
0savev2_adam_dense_787_bias_v_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_dense_784_kernel_read_readvariableop)savev2_dense_784_bias_read_readvariableop+savev2_dense_785_kernel_read_readvariableop)savev2_dense_785_bias_read_readvariableop+savev2_dense_786_kernel_read_readvariableop)savev2_dense_786_bias_read_readvariableop+savev2_dense_787_kernel_read_readvariableop)savev2_dense_787_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop2savev2_adam_dense_784_kernel_m_read_readvariableop0savev2_adam_dense_784_bias_m_read_readvariableop2savev2_adam_dense_785_kernel_m_read_readvariableop0savev2_adam_dense_785_bias_m_read_readvariableop2savev2_adam_dense_786_kernel_m_read_readvariableop0savev2_adam_dense_786_bias_m_read_readvariableop2savev2_adam_dense_787_kernel_m_read_readvariableop0savev2_adam_dense_787_bias_m_read_readvariableop2savev2_adam_dense_784_kernel_v_read_readvariableop0savev2_adam_dense_784_bias_v_read_readvariableop2savev2_adam_dense_785_kernel_v_read_readvariableop0savev2_adam_dense_785_bias_v_read_readvariableop2savev2_adam_dense_786_kernel_v_read_readvariableop0savev2_adam_dense_786_bias_v_read_readvariableop2savev2_adam_dense_787_kernel_v_read_readvariableop0savev2_adam_dense_787_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
?: ::::::::: : : : : : : ::::::::::::::::: 2(
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

:: 

_output_shapes
::$ 

_output_shapes

:: 
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

:: 

_output_shapes
::$ 

_output_shapes

:: 
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

:: 

_output_shapes
::$ 

_output_shapes

:: 
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
?%
?
G__inference_model_196_layer_call_and_return_conditional_losses_18611753

inputs:
(dense_784_matmul_readvariableop_resource:7
)dense_784_biasadd_readvariableop_resource::
(dense_785_matmul_readvariableop_resource:7
)dense_785_biasadd_readvariableop_resource::
(dense_786_matmul_readvariableop_resource:7
)dense_786_biasadd_readvariableop_resource::
(dense_787_matmul_readvariableop_resource:7
)dense_787_biasadd_readvariableop_resource:
identity?? dense_784/BiasAdd/ReadVariableOp?dense_784/MatMul/ReadVariableOp? dense_785/BiasAdd/ReadVariableOp?dense_785/MatMul/ReadVariableOp? dense_786/BiasAdd/ReadVariableOp?dense_786/MatMul/ReadVariableOp? dense_787/BiasAdd/ReadVariableOp?dense_787/MatMul/ReadVariableOp?
dense_784/MatMul/ReadVariableOpReadVariableOp(dense_784_matmul_readvariableop_resource*
_output_shapes

:*
dtype0}
dense_784/MatMulMatMulinputs'dense_784/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
 dense_784/BiasAdd/ReadVariableOpReadVariableOp)dense_784_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_784/BiasAddBiasAdddense_784/MatMul:product:0(dense_784/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d
dense_784/ReluReludense_784/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
dense_785/MatMul/ReadVariableOpReadVariableOp(dense_785_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
dense_785/MatMulMatMuldense_784/Relu:activations:0'dense_785/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
 dense_785/BiasAdd/ReadVariableOpReadVariableOp)dense_785_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_785/BiasAddBiasAdddense_785/MatMul:product:0(dense_785/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d
dense_785/ReluReludense_785/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
dense_786/MatMul/ReadVariableOpReadVariableOp(dense_786_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
dense_786/MatMulMatMuldense_785/Relu:activations:0'dense_786/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
 dense_786/BiasAdd/ReadVariableOpReadVariableOp)dense_786_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_786/BiasAddBiasAdddense_786/MatMul:product:0(dense_786/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d
dense_786/ReluReludense_786/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
dense_787/MatMul/ReadVariableOpReadVariableOp(dense_787_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
dense_787/MatMulMatMuldense_786/Relu:activations:0'dense_787/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
 dense_787/BiasAdd/ReadVariableOpReadVariableOp)dense_787_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_787/BiasAddBiasAdddense_787/MatMul:product:0(dense_787/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????j
dense_787/SigmoidSigmoiddense_787/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d
IdentityIdentitydense_787/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^dense_784/BiasAdd/ReadVariableOp ^dense_784/MatMul/ReadVariableOp!^dense_785/BiasAdd/ReadVariableOp ^dense_785/MatMul/ReadVariableOp!^dense_786/BiasAdd/ReadVariableOp ^dense_786/MatMul/ReadVariableOp!^dense_787/BiasAdd/ReadVariableOp ^dense_787/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 2D
 dense_784/BiasAdd/ReadVariableOp dense_784/BiasAdd/ReadVariableOp2B
dense_784/MatMul/ReadVariableOpdense_784/MatMul/ReadVariableOp2D
 dense_785/BiasAdd/ReadVariableOp dense_785/BiasAdd/ReadVariableOp2B
dense_785/MatMul/ReadVariableOpdense_785/MatMul/ReadVariableOp2D
 dense_786/BiasAdd/ReadVariableOp dense_786/BiasAdd/ReadVariableOp2B
dense_786/MatMul/ReadVariableOpdense_786/MatMul/ReadVariableOp2D
 dense_787/BiasAdd/ReadVariableOp dense_787/BiasAdd/ReadVariableOp2B
dense_787/MatMul/ReadVariableOpdense_787/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
G__inference_dense_787_layer_call_and_return_conditional_losses_18611833

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
?	
?
,__inference_model_196_layer_call_fn_18611689

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
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
G__inference_model_196_layer_call_and_return_conditional_losses_18611530o
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
?
,__inference_model_196_layer_call_fn_18611570
	input_197
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall	input_197unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
G__inference_model_196_layer_call_and_return_conditional_losses_18611530o
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
_user_specified_name	input_197
?%
?
G__inference_model_196_layer_call_and_return_conditional_losses_18611721

inputs:
(dense_784_matmul_readvariableop_resource:7
)dense_784_biasadd_readvariableop_resource::
(dense_785_matmul_readvariableop_resource:7
)dense_785_biasadd_readvariableop_resource::
(dense_786_matmul_readvariableop_resource:7
)dense_786_biasadd_readvariableop_resource::
(dense_787_matmul_readvariableop_resource:7
)dense_787_biasadd_readvariableop_resource:
identity?? dense_784/BiasAdd/ReadVariableOp?dense_784/MatMul/ReadVariableOp? dense_785/BiasAdd/ReadVariableOp?dense_785/MatMul/ReadVariableOp? dense_786/BiasAdd/ReadVariableOp?dense_786/MatMul/ReadVariableOp? dense_787/BiasAdd/ReadVariableOp?dense_787/MatMul/ReadVariableOp?
dense_784/MatMul/ReadVariableOpReadVariableOp(dense_784_matmul_readvariableop_resource*
_output_shapes

:*
dtype0}
dense_784/MatMulMatMulinputs'dense_784/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
 dense_784/BiasAdd/ReadVariableOpReadVariableOp)dense_784_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_784/BiasAddBiasAdddense_784/MatMul:product:0(dense_784/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d
dense_784/ReluReludense_784/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
dense_785/MatMul/ReadVariableOpReadVariableOp(dense_785_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
dense_785/MatMulMatMuldense_784/Relu:activations:0'dense_785/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
 dense_785/BiasAdd/ReadVariableOpReadVariableOp)dense_785_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_785/BiasAddBiasAdddense_785/MatMul:product:0(dense_785/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d
dense_785/ReluReludense_785/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
dense_786/MatMul/ReadVariableOpReadVariableOp(dense_786_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
dense_786/MatMulMatMuldense_785/Relu:activations:0'dense_786/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
 dense_786/BiasAdd/ReadVariableOpReadVariableOp)dense_786_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_786/BiasAddBiasAdddense_786/MatMul:product:0(dense_786/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d
dense_786/ReluReludense_786/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
dense_787/MatMul/ReadVariableOpReadVariableOp(dense_787_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
dense_787/MatMulMatMuldense_786/Relu:activations:0'dense_787/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
 dense_787/BiasAdd/ReadVariableOpReadVariableOp)dense_787_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_787/BiasAddBiasAdddense_787/MatMul:product:0(dense_787/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????j
dense_787/SigmoidSigmoiddense_787/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d
IdentityIdentitydense_787/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^dense_784/BiasAdd/ReadVariableOp ^dense_784/MatMul/ReadVariableOp!^dense_785/BiasAdd/ReadVariableOp ^dense_785/MatMul/ReadVariableOp!^dense_786/BiasAdd/ReadVariableOp ^dense_786/MatMul/ReadVariableOp!^dense_787/BiasAdd/ReadVariableOp ^dense_787/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 2D
 dense_784/BiasAdd/ReadVariableOp dense_784/BiasAdd/ReadVariableOp2B
dense_784/MatMul/ReadVariableOpdense_784/MatMul/ReadVariableOp2D
 dense_785/BiasAdd/ReadVariableOp dense_785/BiasAdd/ReadVariableOp2B
dense_785/MatMul/ReadVariableOpdense_785/MatMul/ReadVariableOp2D
 dense_786/BiasAdd/ReadVariableOp dense_786/BiasAdd/ReadVariableOp2B
dense_786/MatMul/ReadVariableOpdense_786/MatMul/ReadVariableOp2D
 dense_787/BiasAdd/ReadVariableOp dense_787/BiasAdd/ReadVariableOp2B
dense_787/MatMul/ReadVariableOpdense_787/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
G__inference_dense_786_layer_call_and_return_conditional_losses_18611400

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
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
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
,__inference_model_196_layer_call_fn_18611443
	input_197
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall	input_197unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
G__inference_model_196_layer_call_and_return_conditional_losses_18611424o
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
_user_specified_name	input_197
?	
?
&__inference_signature_wrapper_18611647
	input_197
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall	input_197unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
#__inference__wrapped_model_18611348o
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
_user_specified_name	input_197
?

?
G__inference_dense_784_layer_call_and_return_conditional_losses_18611366

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
,__inference_dense_785_layer_call_fn_18611782

inputs
unknown:
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
G__inference_dense_785_layer_call_and_return_conditional_losses_18611383o
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
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
G__inference_dense_785_layer_call_and_return_conditional_losses_18611793

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
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
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
,__inference_dense_786_layer_call_fn_18611802

inputs
unknown:
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
G__inference_dense_786_layer_call_and_return_conditional_losses_18611400o
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
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
G__inference_dense_787_layer_call_and_return_conditional_losses_18611417

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
G__inference_model_196_layer_call_and_return_conditional_losses_18611618
	input_197$
dense_784_18611597: 
dense_784_18611599:$
dense_785_18611602: 
dense_785_18611604:$
dense_786_18611607: 
dense_786_18611609:$
dense_787_18611612: 
dense_787_18611614:
identity??!dense_784/StatefulPartitionedCall?!dense_785/StatefulPartitionedCall?!dense_786/StatefulPartitionedCall?!dense_787/StatefulPartitionedCall?
!dense_784/StatefulPartitionedCallStatefulPartitionedCall	input_197dense_784_18611597dense_784_18611599*
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
G__inference_dense_784_layer_call_and_return_conditional_losses_18611366?
!dense_785/StatefulPartitionedCallStatefulPartitionedCall*dense_784/StatefulPartitionedCall:output:0dense_785_18611602dense_785_18611604*
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
G__inference_dense_785_layer_call_and_return_conditional_losses_18611383?
!dense_786/StatefulPartitionedCallStatefulPartitionedCall*dense_785/StatefulPartitionedCall:output:0dense_786_18611607dense_786_18611609*
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
G__inference_dense_786_layer_call_and_return_conditional_losses_18611400?
!dense_787/StatefulPartitionedCallStatefulPartitionedCall*dense_786/StatefulPartitionedCall:output:0dense_787_18611612dense_787_18611614*
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
G__inference_dense_787_layer_call_and_return_conditional_losses_18611417y
IdentityIdentity*dense_787/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp"^dense_784/StatefulPartitionedCall"^dense_785/StatefulPartitionedCall"^dense_786/StatefulPartitionedCall"^dense_787/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 2F
!dense_784/StatefulPartitionedCall!dense_784/StatefulPartitionedCall2F
!dense_785/StatefulPartitionedCall!dense_785/StatefulPartitionedCall2F
!dense_786/StatefulPartitionedCall!dense_786/StatefulPartitionedCall2F
!dense_787/StatefulPartitionedCall!dense_787/StatefulPartitionedCall:R N
'
_output_shapes
:?????????
#
_user_specified_name	input_197
?

?
G__inference_dense_786_layer_call_and_return_conditional_losses_18611813

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
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
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?+
?
#__inference__wrapped_model_18611348
	input_197D
2model_196_dense_784_matmul_readvariableop_resource:A
3model_196_dense_784_biasadd_readvariableop_resource:D
2model_196_dense_785_matmul_readvariableop_resource:A
3model_196_dense_785_biasadd_readvariableop_resource:D
2model_196_dense_786_matmul_readvariableop_resource:A
3model_196_dense_786_biasadd_readvariableop_resource:D
2model_196_dense_787_matmul_readvariableop_resource:A
3model_196_dense_787_biasadd_readvariableop_resource:
identity??*model_196/dense_784/BiasAdd/ReadVariableOp?)model_196/dense_784/MatMul/ReadVariableOp?*model_196/dense_785/BiasAdd/ReadVariableOp?)model_196/dense_785/MatMul/ReadVariableOp?*model_196/dense_786/BiasAdd/ReadVariableOp?)model_196/dense_786/MatMul/ReadVariableOp?*model_196/dense_787/BiasAdd/ReadVariableOp?)model_196/dense_787/MatMul/ReadVariableOp?
)model_196/dense_784/MatMul/ReadVariableOpReadVariableOp2model_196_dense_784_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
model_196/dense_784/MatMulMatMul	input_1971model_196/dense_784/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
*model_196/dense_784/BiasAdd/ReadVariableOpReadVariableOp3model_196_dense_784_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model_196/dense_784/BiasAddBiasAdd$model_196/dense_784/MatMul:product:02model_196/dense_784/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????x
model_196/dense_784/ReluRelu$model_196/dense_784/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
)model_196/dense_785/MatMul/ReadVariableOpReadVariableOp2model_196_dense_785_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
model_196/dense_785/MatMulMatMul&model_196/dense_784/Relu:activations:01model_196/dense_785/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
*model_196/dense_785/BiasAdd/ReadVariableOpReadVariableOp3model_196_dense_785_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model_196/dense_785/BiasAddBiasAdd$model_196/dense_785/MatMul:product:02model_196/dense_785/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????x
model_196/dense_785/ReluRelu$model_196/dense_785/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
)model_196/dense_786/MatMul/ReadVariableOpReadVariableOp2model_196_dense_786_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
model_196/dense_786/MatMulMatMul&model_196/dense_785/Relu:activations:01model_196/dense_786/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
*model_196/dense_786/BiasAdd/ReadVariableOpReadVariableOp3model_196_dense_786_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model_196/dense_786/BiasAddBiasAdd$model_196/dense_786/MatMul:product:02model_196/dense_786/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????x
model_196/dense_786/ReluRelu$model_196/dense_786/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
)model_196/dense_787/MatMul/ReadVariableOpReadVariableOp2model_196_dense_787_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
model_196/dense_787/MatMulMatMul&model_196/dense_786/Relu:activations:01model_196/dense_787/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
*model_196/dense_787/BiasAdd/ReadVariableOpReadVariableOp3model_196_dense_787_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model_196/dense_787/BiasAddBiasAdd$model_196/dense_787/MatMul:product:02model_196/dense_787/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????~
model_196/dense_787/SigmoidSigmoid$model_196/dense_787/BiasAdd:output:0*
T0*'
_output_shapes
:?????????n
IdentityIdentitymodel_196/dense_787/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp+^model_196/dense_784/BiasAdd/ReadVariableOp*^model_196/dense_784/MatMul/ReadVariableOp+^model_196/dense_785/BiasAdd/ReadVariableOp*^model_196/dense_785/MatMul/ReadVariableOp+^model_196/dense_786/BiasAdd/ReadVariableOp*^model_196/dense_786/MatMul/ReadVariableOp+^model_196/dense_787/BiasAdd/ReadVariableOp*^model_196/dense_787/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 2X
*model_196/dense_784/BiasAdd/ReadVariableOp*model_196/dense_784/BiasAdd/ReadVariableOp2V
)model_196/dense_784/MatMul/ReadVariableOp)model_196/dense_784/MatMul/ReadVariableOp2X
*model_196/dense_785/BiasAdd/ReadVariableOp*model_196/dense_785/BiasAdd/ReadVariableOp2V
)model_196/dense_785/MatMul/ReadVariableOp)model_196/dense_785/MatMul/ReadVariableOp2X
*model_196/dense_786/BiasAdd/ReadVariableOp*model_196/dense_786/BiasAdd/ReadVariableOp2V
)model_196/dense_786/MatMul/ReadVariableOp)model_196/dense_786/MatMul/ReadVariableOp2X
*model_196/dense_787/BiasAdd/ReadVariableOp*model_196/dense_787/BiasAdd/ReadVariableOp2V
)model_196/dense_787/MatMul/ReadVariableOp)model_196/dense_787/MatMul/ReadVariableOp:R N
'
_output_shapes
:?????????
#
_user_specified_name	input_197
?
?
G__inference_model_196_layer_call_and_return_conditional_losses_18611594
	input_197$
dense_784_18611573: 
dense_784_18611575:$
dense_785_18611578: 
dense_785_18611580:$
dense_786_18611583: 
dense_786_18611585:$
dense_787_18611588: 
dense_787_18611590:
identity??!dense_784/StatefulPartitionedCall?!dense_785/StatefulPartitionedCall?!dense_786/StatefulPartitionedCall?!dense_787/StatefulPartitionedCall?
!dense_784/StatefulPartitionedCallStatefulPartitionedCall	input_197dense_784_18611573dense_784_18611575*
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
G__inference_dense_784_layer_call_and_return_conditional_losses_18611366?
!dense_785/StatefulPartitionedCallStatefulPartitionedCall*dense_784/StatefulPartitionedCall:output:0dense_785_18611578dense_785_18611580*
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
G__inference_dense_785_layer_call_and_return_conditional_losses_18611383?
!dense_786/StatefulPartitionedCallStatefulPartitionedCall*dense_785/StatefulPartitionedCall:output:0dense_786_18611583dense_786_18611585*
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
G__inference_dense_786_layer_call_and_return_conditional_losses_18611400?
!dense_787/StatefulPartitionedCallStatefulPartitionedCall*dense_786/StatefulPartitionedCall:output:0dense_787_18611588dense_787_18611590*
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
G__inference_dense_787_layer_call_and_return_conditional_losses_18611417y
IdentityIdentity*dense_787/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp"^dense_784/StatefulPartitionedCall"^dense_785/StatefulPartitionedCall"^dense_786/StatefulPartitionedCall"^dense_787/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 2F
!dense_784/StatefulPartitionedCall!dense_784/StatefulPartitionedCall2F
!dense_785/StatefulPartitionedCall!dense_785/StatefulPartitionedCall2F
!dense_786/StatefulPartitionedCall!dense_786/StatefulPartitionedCall2F
!dense_787/StatefulPartitionedCall!dense_787/StatefulPartitionedCall:R N
'
_output_shapes
:?????????
#
_user_specified_name	input_197
?
?
,__inference_dense_787_layer_call_fn_18611822

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
G__inference_dense_787_layer_call_and_return_conditional_losses_18611417o
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
?
,__inference_model_196_layer_call_fn_18611668

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
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
G__inference_model_196_layer_call_and_return_conditional_losses_18611424o
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
	input_1972
serving_default_input_197:0?????????=
	dense_7870
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
": 2dense_784/kernel
:2dense_784/bias
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
": 2dense_785/kernel
:2dense_785/bias
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
": 2dense_786/kernel
:2dense_786/bias
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
": 2dense_787/kernel
:2dense_787/bias
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
':%2Adam/dense_784/kernel/m
!:2Adam/dense_784/bias/m
':%2Adam/dense_785/kernel/m
!:2Adam/dense_785/bias/m
':%2Adam/dense_786/kernel/m
!:2Adam/dense_786/bias/m
':%2Adam/dense_787/kernel/m
!:2Adam/dense_787/bias/m
':%2Adam/dense_784/kernel/v
!:2Adam/dense_784/bias/v
':%2Adam/dense_785/kernel/v
!:2Adam/dense_785/bias/v
':%2Adam/dense_786/kernel/v
!:2Adam/dense_786/bias/v
':%2Adam/dense_787/kernel/v
!:2Adam/dense_787/bias/v
?2?
,__inference_model_196_layer_call_fn_18611443
,__inference_model_196_layer_call_fn_18611668
,__inference_model_196_layer_call_fn_18611689
,__inference_model_196_layer_call_fn_18611570?
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
G__inference_model_196_layer_call_and_return_conditional_losses_18611721
G__inference_model_196_layer_call_and_return_conditional_losses_18611753
G__inference_model_196_layer_call_and_return_conditional_losses_18611594
G__inference_model_196_layer_call_and_return_conditional_losses_18611618?
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
#__inference__wrapped_model_18611348	input_197"?
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
,__inference_dense_784_layer_call_fn_18611762?
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
G__inference_dense_784_layer_call_and_return_conditional_losses_18611773?
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
,__inference_dense_785_layer_call_fn_18611782?
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
G__inference_dense_785_layer_call_and_return_conditional_losses_18611793?
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
,__inference_dense_786_layer_call_fn_18611802?
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
G__inference_dense_786_layer_call_and_return_conditional_losses_18611813?
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
,__inference_dense_787_layer_call_fn_18611822?
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
G__inference_dense_787_layer_call_and_return_conditional_losses_18611833?
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
&__inference_signature_wrapper_18611647	input_197"?
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
#__inference__wrapped_model_18611348u2?/
(?%
#? 
	input_197?????????
? "5?2
0
	dense_787#? 
	dense_787??????????
G__inference_dense_784_layer_call_and_return_conditional_losses_18611773\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? 
,__inference_dense_784_layer_call_fn_18611762O/?,
%?"
 ?
inputs?????????
? "???????????
G__inference_dense_785_layer_call_and_return_conditional_losses_18611793\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? 
,__inference_dense_785_layer_call_fn_18611782O/?,
%?"
 ?
inputs?????????
? "???????????
G__inference_dense_786_layer_call_and_return_conditional_losses_18611813\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? 
,__inference_dense_786_layer_call_fn_18611802O/?,
%?"
 ?
inputs?????????
? "???????????
G__inference_dense_787_layer_call_and_return_conditional_losses_18611833\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? 
,__inference_dense_787_layer_call_fn_18611822O/?,
%?"
 ?
inputs?????????
? "???????????
G__inference_model_196_layer_call_and_return_conditional_losses_18611594m:?7
0?-
#? 
	input_197?????????
p 

 
? "%?"
?
0?????????
? ?
G__inference_model_196_layer_call_and_return_conditional_losses_18611618m:?7
0?-
#? 
	input_197?????????
p

 
? "%?"
?
0?????????
? ?
G__inference_model_196_layer_call_and_return_conditional_losses_18611721j7?4
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
G__inference_model_196_layer_call_and_return_conditional_losses_18611753j7?4
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
,__inference_model_196_layer_call_fn_18611443`:?7
0?-
#? 
	input_197?????????
p 

 
? "???????????
,__inference_model_196_layer_call_fn_18611570`:?7
0?-
#? 
	input_197?????????
p

 
? "???????????
,__inference_model_196_layer_call_fn_18611668]7?4
-?*
 ?
inputs?????????
p 

 
? "???????????
,__inference_model_196_layer_call_fn_18611689]7?4
-?*
 ?
inputs?????????
p

 
? "???????????
&__inference_signature_wrapper_18611647???<
? 
5?2
0
	input_197#? 
	input_197?????????"5?2
0
	dense_787#? 
	dense_787?????????
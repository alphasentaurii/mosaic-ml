??'
??
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
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv3D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)(0""
paddingstring:
SAMEVALID"0
data_formatstringNDHWC:
NDHWCNCDHW"!
	dilations	list(int)	

?
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%??8"&
exponential_avg_factorfloat%  ??";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
\
	LeakyRelu
features"T
activations"T"
alphafloat%??L>"
Ttype0:
2
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
	MaxPool3D

input"T
output"T"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"0
data_formatstringNDHWC:
NDHWCNCDHW"
Ttype:
2
?
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
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
executor_typestring ?
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
 ?"serve*2.6.02v2.6.0-0-g919f693420e8??#
?
conv3d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv3d_5/kernel

#conv3d_5/kernel/Read/ReadVariableOpReadVariableOpconv3d_5/kernel**
_output_shapes
: *
dtype0
r
conv3d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv3d_5/bias
k
!conv3d_5/bias/Read/ReadVariableOpReadVariableOpconv3d_5/bias*
_output_shapes
: *
dtype0
?
batch_normalization_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_5/gamma
?
/batch_normalization_5/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_5/gamma*
_output_shapes
: *
dtype0
?
batch_normalization_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namebatch_normalization_5/beta
?
.batch_normalization_5/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_5/beta*
_output_shapes
: *
dtype0
?
!batch_normalization_5/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!batch_normalization_5/moving_mean
?
5batch_normalization_5/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_5/moving_mean*
_output_shapes
: *
dtype0
?
%batch_normalization_5/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%batch_normalization_5/moving_variance
?
9batch_normalization_5/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_5/moving_variance*
_output_shapes
: *
dtype0
?
conv3d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  * 
shared_nameconv3d_6/kernel

#conv3d_6/kernel/Read/ReadVariableOpReadVariableOpconv3d_6/kernel**
_output_shapes
:  *
dtype0
r
conv3d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv3d_6/bias
k
!conv3d_6/bias/Read/ReadVariableOpReadVariableOpconv3d_6/bias*
_output_shapes
: *
dtype0
?
batch_normalization_6/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_6/gamma
?
/batch_normalization_6/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_6/gamma*
_output_shapes
: *
dtype0
?
batch_normalization_6/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namebatch_normalization_6/beta
?
.batch_normalization_6/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_6/beta*
_output_shapes
: *
dtype0
?
!batch_normalization_6/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!batch_normalization_6/moving_mean
?
5batch_normalization_6/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_6/moving_mean*
_output_shapes
: *
dtype0
?
%batch_normalization_6/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%batch_normalization_6/moving_variance
?
9batch_normalization_6/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_6/moving_variance*
_output_shapes
: *
dtype0
?
conv3d_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @* 
shared_nameconv3d_7/kernel

#conv3d_7/kernel/Read/ReadVariableOpReadVariableOpconv3d_7/kernel**
_output_shapes
: @*
dtype0
r
conv3d_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv3d_7/bias
k
!conv3d_7/bias/Read/ReadVariableOpReadVariableOpconv3d_7/bias*
_output_shapes
:@*
dtype0
?
batch_normalization_7/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_7/gamma
?
/batch_normalization_7/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_7/gamma*
_output_shapes
:@*
dtype0
?
batch_normalization_7/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_7/beta
?
.batch_normalization_7/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_7/beta*
_output_shapes
:@*
dtype0
?
!batch_normalization_7/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_7/moving_mean
?
5batch_normalization_7/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_7/moving_mean*
_output_shapes
:@*
dtype0
?
%batch_normalization_7/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_7/moving_variance
?
9batch_normalization_7/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_7/moving_variance*
_output_shapes
:@*
dtype0
?
conv3d_8/kernelVarHandleOp*
_output_shapes
: *
dtype0* 
shape:@?* 
shared_nameconv3d_8/kernel
?
#conv3d_8/kernel/Read/ReadVariableOpReadVariableOpconv3d_8/kernel*+
_output_shapes
:@?*
dtype0
s
conv3d_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv3d_8/bias
l
!conv3d_8/bias/Read/ReadVariableOpReadVariableOpconv3d_8/bias*
_output_shapes	
:?*
dtype0
?
batch_normalization_8/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_namebatch_normalization_8/gamma
?
/batch_normalization_8/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_8/gamma*
_output_shapes	
:?*
dtype0
?
batch_normalization_8/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*+
shared_namebatch_normalization_8/beta
?
.batch_normalization_8/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_8/beta*
_output_shapes	
:?*
dtype0
?
!batch_normalization_8/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*2
shared_name#!batch_normalization_8/moving_mean
?
5batch_normalization_8/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_8/moving_mean*
_output_shapes	
:?*
dtype0
?
%batch_normalization_8/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*6
shared_name'%batch_normalization_8/moving_variance
?
9batch_normalization_8/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_8/moving_variance*
_output_shapes	
:?*
dtype0
?
conv3d_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*!
shape:??* 
shared_nameconv3d_9/kernel
?
#conv3d_9/kernel/Read/ReadVariableOpReadVariableOpconv3d_9/kernel*,
_output_shapes
:??*
dtype0
s
conv3d_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv3d_9/bias
l
!conv3d_9/bias/Read/ReadVariableOpReadVariableOpconv3d_9/bias*
_output_shapes	
:?*
dtype0
|
1_dense18/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*!
shared_name1_dense18/kernel
u
$1_dense18/kernel/Read/ReadVariableOpReadVariableOp1_dense18/kernel*
_output_shapes

:
*
dtype0
t
1_dense18/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name1_dense18/bias
m
"1_dense18/bias/Read/ReadVariableOpReadVariableOp1_dense18/bias*
_output_shapes
:*
dtype0
|
2_dense32/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *!
shared_name2_dense32/kernel
u
$2_dense32/kernel/Read/ReadVariableOpReadVariableOp2_dense32/kernel*
_output_shapes

: *
dtype0
t
2_dense32/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name2_dense32/bias
m
"2_dense32/bias/Read/ReadVariableOpReadVariableOp2_dense32/bias*
_output_shapes
: *
dtype0
?
batch_normalization_9/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_namebatch_normalization_9/gamma
?
/batch_normalization_9/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_9/gamma*
_output_shapes	
:?*
dtype0
?
batch_normalization_9/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*+
shared_namebatch_normalization_9/beta
?
.batch_normalization_9/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_9/beta*
_output_shapes	
:?*
dtype0
?
!batch_normalization_9/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*2
shared_name#!batch_normalization_9/moving_mean
?
5batch_normalization_9/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_9/moving_mean*
_output_shapes	
:?*
dtype0
?
%batch_normalization_9/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*6
shared_name'%batch_normalization_9/moving_variance
?
9batch_normalization_9/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_9/moving_variance*
_output_shapes	
:?*
dtype0
|
3_dense64/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: @*!
shared_name3_dense64/kernel
u
$3_dense64/kernel/Read/ReadVariableOpReadVariableOp3_dense64/kernel*
_output_shapes

: @*
dtype0
t
3_dense64/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name3_dense64/bias
m
"3_dense64/bias/Read/ReadVariableOpReadVariableOp3_dense64/bias*
_output_shapes
:@*
dtype0
|
4_dense32/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *!
shared_name4_dense32/kernel
u
$4_dense32/kernel/Read/ReadVariableOpReadVariableOp4_dense32/kernel*
_output_shapes

:@ *
dtype0
t
4_dense32/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name4_dense32/bias
m
"4_dense32/bias/Read/ReadVariableOpReadVariableOp4_dense32/bias*
_output_shapes
: *
dtype0
z
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_namedense_1/kernel
s
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel* 
_output_shapes
:
??*
dtype0
q
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_1/bias
j
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes	
:?*
dtype0
|
5_dense18/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *!
shared_name5_dense18/kernel
u
$5_dense18/kernel/Read/ReadVariableOpReadVariableOp5_dense18/kernel*
_output_shapes

: *
dtype0
t
5_dense18/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name5_dense18/bias
m
"5_dense18/bias/Read/ReadVariableOpReadVariableOp5_dense18/bias*
_output_shapes
:*
dtype0
?
combined_input/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?	*&
shared_namecombined_input/kernel
?
)combined_input/kernel/Read/ReadVariableOpReadVariableOpcombined_input/kernel*
_output_shapes
:	?	*
dtype0
~
combined_input/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*$
shared_namecombined_input/bias
w
'combined_input/bias/Read/ReadVariableOpReadVariableOpcombined_input/bias*
_output_shapes
:	*
dtype0
?
ensemble_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	*'
shared_nameensemble_output/kernel
?
*ensemble_output/kernel/Read/ReadVariableOpReadVariableOpensemble_output/kernel*
_output_shapes

:	*
dtype0
?
ensemble_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameensemble_output/bias
y
(ensemble_output/bias/Read/ReadVariableOpReadVariableOpensemble_output/bias*
_output_shapes
:*
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
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
Adam/conv3d_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv3d_5/kernel/m
?
*Adam/conv3d_5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_5/kernel/m**
_output_shapes
: *
dtype0
?
Adam/conv3d_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv3d_5/bias/m
y
(Adam/conv3d_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_5/bias/m*
_output_shapes
: *
dtype0
?
"Adam/batch_normalization_5/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_5/gamma/m
?
6Adam/batch_normalization_5/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_5/gamma/m*
_output_shapes
: *
dtype0
?
!Adam/batch_normalization_5/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/batch_normalization_5/beta/m
?
5Adam/batch_normalization_5/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_5/beta/m*
_output_shapes
: *
dtype0
?
Adam/conv3d_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *'
shared_nameAdam/conv3d_6/kernel/m
?
*Adam/conv3d_6/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_6/kernel/m**
_output_shapes
:  *
dtype0
?
Adam/conv3d_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv3d_6/bias/m
y
(Adam/conv3d_6/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_6/bias/m*
_output_shapes
: *
dtype0
?
"Adam/batch_normalization_6/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_6/gamma/m
?
6Adam/batch_normalization_6/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_6/gamma/m*
_output_shapes
: *
dtype0
?
!Adam/batch_normalization_6/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/batch_normalization_6/beta/m
?
5Adam/batch_normalization_6/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_6/beta/m*
_output_shapes
: *
dtype0
?
Adam/conv3d_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*'
shared_nameAdam/conv3d_7/kernel/m
?
*Adam/conv3d_7/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_7/kernel/m**
_output_shapes
: @*
dtype0
?
Adam/conv3d_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv3d_7/bias/m
y
(Adam/conv3d_7/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_7/bias/m*
_output_shapes
:@*
dtype0
?
"Adam/batch_normalization_7/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_7/gamma/m
?
6Adam/batch_normalization_7/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_7/gamma/m*
_output_shapes
:@*
dtype0
?
!Adam/batch_normalization_7/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/batch_normalization_7/beta/m
?
5Adam/batch_normalization_7/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_7/beta/m*
_output_shapes
:@*
dtype0
?
Adam/conv3d_8/kernel/mVarHandleOp*
_output_shapes
: *
dtype0* 
shape:@?*'
shared_nameAdam/conv3d_8/kernel/m
?
*Adam/conv3d_8/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_8/kernel/m*+
_output_shapes
:@?*
dtype0
?
Adam/conv3d_8/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/conv3d_8/bias/m
z
(Adam/conv3d_8/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_8/bias/m*
_output_shapes	
:?*
dtype0
?
"Adam/batch_normalization_8/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"Adam/batch_normalization_8/gamma/m
?
6Adam/batch_normalization_8/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_8/gamma/m*
_output_shapes	
:?*
dtype0
?
!Adam/batch_normalization_8/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*2
shared_name#!Adam/batch_normalization_8/beta/m
?
5Adam/batch_normalization_8/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_8/beta/m*
_output_shapes	
:?*
dtype0
?
Adam/conv3d_9/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*!
shape:??*'
shared_nameAdam/conv3d_9/kernel/m
?
*Adam/conv3d_9/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_9/kernel/m*,
_output_shapes
:??*
dtype0
?
Adam/conv3d_9/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/conv3d_9/bias/m
z
(Adam/conv3d_9/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_9/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/1_dense18/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*(
shared_nameAdam/1_dense18/kernel/m
?
+Adam/1_dense18/kernel/m/Read/ReadVariableOpReadVariableOpAdam/1_dense18/kernel/m*
_output_shapes

:
*
dtype0
?
Adam/1_dense18/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/1_dense18/bias/m
{
)Adam/1_dense18/bias/m/Read/ReadVariableOpReadVariableOpAdam/1_dense18/bias/m*
_output_shapes
:*
dtype0
?
Adam/2_dense32/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *(
shared_nameAdam/2_dense32/kernel/m
?
+Adam/2_dense32/kernel/m/Read/ReadVariableOpReadVariableOpAdam/2_dense32/kernel/m*
_output_shapes

: *
dtype0
?
Adam/2_dense32/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/2_dense32/bias/m
{
)Adam/2_dense32/bias/m/Read/ReadVariableOpReadVariableOpAdam/2_dense32/bias/m*
_output_shapes
: *
dtype0
?
"Adam/batch_normalization_9/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"Adam/batch_normalization_9/gamma/m
?
6Adam/batch_normalization_9/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_9/gamma/m*
_output_shapes	
:?*
dtype0
?
!Adam/batch_normalization_9/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*2
shared_name#!Adam/batch_normalization_9/beta/m
?
5Adam/batch_normalization_9/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_9/beta/m*
_output_shapes	
:?*
dtype0
?
Adam/3_dense64/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: @*(
shared_nameAdam/3_dense64/kernel/m
?
+Adam/3_dense64/kernel/m/Read/ReadVariableOpReadVariableOpAdam/3_dense64/kernel/m*
_output_shapes

: @*
dtype0
?
Adam/3_dense64/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/3_dense64/bias/m
{
)Adam/3_dense64/bias/m/Read/ReadVariableOpReadVariableOpAdam/3_dense64/bias/m*
_output_shapes
:@*
dtype0
?
Adam/4_dense32/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *(
shared_nameAdam/4_dense32/kernel/m
?
+Adam/4_dense32/kernel/m/Read/ReadVariableOpReadVariableOpAdam/4_dense32/kernel/m*
_output_shapes

:@ *
dtype0
?
Adam/4_dense32/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/4_dense32/bias/m
{
)Adam/4_dense32/bias/m/Read/ReadVariableOpReadVariableOpAdam/4_dense32/bias/m*
_output_shapes
: *
dtype0
?
Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*&
shared_nameAdam/dense_1/kernel/m
?
)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m* 
_output_shapes
:
??*
dtype0

Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*$
shared_nameAdam/dense_1/bias/m
x
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/5_dense18/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *(
shared_nameAdam/5_dense18/kernel/m
?
+Adam/5_dense18/kernel/m/Read/ReadVariableOpReadVariableOpAdam/5_dense18/kernel/m*
_output_shapes

: *
dtype0
?
Adam/5_dense18/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/5_dense18/bias/m
{
)Adam/5_dense18/bias/m/Read/ReadVariableOpReadVariableOpAdam/5_dense18/bias/m*
_output_shapes
:*
dtype0
?
Adam/combined_input/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?	*-
shared_nameAdam/combined_input/kernel/m
?
0Adam/combined_input/kernel/m/Read/ReadVariableOpReadVariableOpAdam/combined_input/kernel/m*
_output_shapes
:	?	*
dtype0
?
Adam/combined_input/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*+
shared_nameAdam/combined_input/bias/m
?
.Adam/combined_input/bias/m/Read/ReadVariableOpReadVariableOpAdam/combined_input/bias/m*
_output_shapes
:	*
dtype0
?
Adam/ensemble_output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	*.
shared_nameAdam/ensemble_output/kernel/m
?
1Adam/ensemble_output/kernel/m/Read/ReadVariableOpReadVariableOpAdam/ensemble_output/kernel/m*
_output_shapes

:	*
dtype0
?
Adam/ensemble_output/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameAdam/ensemble_output/bias/m
?
/Adam/ensemble_output/bias/m/Read/ReadVariableOpReadVariableOpAdam/ensemble_output/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv3d_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv3d_5/kernel/v
?
*Adam/conv3d_5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_5/kernel/v**
_output_shapes
: *
dtype0
?
Adam/conv3d_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv3d_5/bias/v
y
(Adam/conv3d_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_5/bias/v*
_output_shapes
: *
dtype0
?
"Adam/batch_normalization_5/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_5/gamma/v
?
6Adam/batch_normalization_5/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_5/gamma/v*
_output_shapes
: *
dtype0
?
!Adam/batch_normalization_5/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/batch_normalization_5/beta/v
?
5Adam/batch_normalization_5/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_5/beta/v*
_output_shapes
: *
dtype0
?
Adam/conv3d_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *'
shared_nameAdam/conv3d_6/kernel/v
?
*Adam/conv3d_6/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_6/kernel/v**
_output_shapes
:  *
dtype0
?
Adam/conv3d_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv3d_6/bias/v
y
(Adam/conv3d_6/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_6/bias/v*
_output_shapes
: *
dtype0
?
"Adam/batch_normalization_6/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_6/gamma/v
?
6Adam/batch_normalization_6/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_6/gamma/v*
_output_shapes
: *
dtype0
?
!Adam/batch_normalization_6/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/batch_normalization_6/beta/v
?
5Adam/batch_normalization_6/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_6/beta/v*
_output_shapes
: *
dtype0
?
Adam/conv3d_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*'
shared_nameAdam/conv3d_7/kernel/v
?
*Adam/conv3d_7/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_7/kernel/v**
_output_shapes
: @*
dtype0
?
Adam/conv3d_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv3d_7/bias/v
y
(Adam/conv3d_7/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_7/bias/v*
_output_shapes
:@*
dtype0
?
"Adam/batch_normalization_7/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_7/gamma/v
?
6Adam/batch_normalization_7/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_7/gamma/v*
_output_shapes
:@*
dtype0
?
!Adam/batch_normalization_7/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/batch_normalization_7/beta/v
?
5Adam/batch_normalization_7/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_7/beta/v*
_output_shapes
:@*
dtype0
?
Adam/conv3d_8/kernel/vVarHandleOp*
_output_shapes
: *
dtype0* 
shape:@?*'
shared_nameAdam/conv3d_8/kernel/v
?
*Adam/conv3d_8/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_8/kernel/v*+
_output_shapes
:@?*
dtype0
?
Adam/conv3d_8/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/conv3d_8/bias/v
z
(Adam/conv3d_8/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_8/bias/v*
_output_shapes	
:?*
dtype0
?
"Adam/batch_normalization_8/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"Adam/batch_normalization_8/gamma/v
?
6Adam/batch_normalization_8/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_8/gamma/v*
_output_shapes	
:?*
dtype0
?
!Adam/batch_normalization_8/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*2
shared_name#!Adam/batch_normalization_8/beta/v
?
5Adam/batch_normalization_8/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_8/beta/v*
_output_shapes	
:?*
dtype0
?
Adam/conv3d_9/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*!
shape:??*'
shared_nameAdam/conv3d_9/kernel/v
?
*Adam/conv3d_9/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_9/kernel/v*,
_output_shapes
:??*
dtype0
?
Adam/conv3d_9/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/conv3d_9/bias/v
z
(Adam/conv3d_9/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_9/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/1_dense18/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*(
shared_nameAdam/1_dense18/kernel/v
?
+Adam/1_dense18/kernel/v/Read/ReadVariableOpReadVariableOpAdam/1_dense18/kernel/v*
_output_shapes

:
*
dtype0
?
Adam/1_dense18/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/1_dense18/bias/v
{
)Adam/1_dense18/bias/v/Read/ReadVariableOpReadVariableOpAdam/1_dense18/bias/v*
_output_shapes
:*
dtype0
?
Adam/2_dense32/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *(
shared_nameAdam/2_dense32/kernel/v
?
+Adam/2_dense32/kernel/v/Read/ReadVariableOpReadVariableOpAdam/2_dense32/kernel/v*
_output_shapes

: *
dtype0
?
Adam/2_dense32/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/2_dense32/bias/v
{
)Adam/2_dense32/bias/v/Read/ReadVariableOpReadVariableOpAdam/2_dense32/bias/v*
_output_shapes
: *
dtype0
?
"Adam/batch_normalization_9/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"Adam/batch_normalization_9/gamma/v
?
6Adam/batch_normalization_9/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_9/gamma/v*
_output_shapes	
:?*
dtype0
?
!Adam/batch_normalization_9/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*2
shared_name#!Adam/batch_normalization_9/beta/v
?
5Adam/batch_normalization_9/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_9/beta/v*
_output_shapes	
:?*
dtype0
?
Adam/3_dense64/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: @*(
shared_nameAdam/3_dense64/kernel/v
?
+Adam/3_dense64/kernel/v/Read/ReadVariableOpReadVariableOpAdam/3_dense64/kernel/v*
_output_shapes

: @*
dtype0
?
Adam/3_dense64/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/3_dense64/bias/v
{
)Adam/3_dense64/bias/v/Read/ReadVariableOpReadVariableOpAdam/3_dense64/bias/v*
_output_shapes
:@*
dtype0
?
Adam/4_dense32/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *(
shared_nameAdam/4_dense32/kernel/v
?
+Adam/4_dense32/kernel/v/Read/ReadVariableOpReadVariableOpAdam/4_dense32/kernel/v*
_output_shapes

:@ *
dtype0
?
Adam/4_dense32/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/4_dense32/bias/v
{
)Adam/4_dense32/bias/v/Read/ReadVariableOpReadVariableOpAdam/4_dense32/bias/v*
_output_shapes
: *
dtype0
?
Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*&
shared_nameAdam/dense_1/kernel/v
?
)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v* 
_output_shapes
:
??*
dtype0

Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*$
shared_nameAdam/dense_1/bias/v
x
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/5_dense18/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *(
shared_nameAdam/5_dense18/kernel/v
?
+Adam/5_dense18/kernel/v/Read/ReadVariableOpReadVariableOpAdam/5_dense18/kernel/v*
_output_shapes

: *
dtype0
?
Adam/5_dense18/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/5_dense18/bias/v
{
)Adam/5_dense18/bias/v/Read/ReadVariableOpReadVariableOpAdam/5_dense18/bias/v*
_output_shapes
:*
dtype0
?
Adam/combined_input/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?	*-
shared_nameAdam/combined_input/kernel/v
?
0Adam/combined_input/kernel/v/Read/ReadVariableOpReadVariableOpAdam/combined_input/kernel/v*
_output_shapes
:	?	*
dtype0
?
Adam/combined_input/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*+
shared_nameAdam/combined_input/bias/v
?
.Adam/combined_input/bias/v/Read/ReadVariableOpReadVariableOpAdam/combined_input/bias/v*
_output_shapes
:	*
dtype0
?
Adam/ensemble_output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	*.
shared_nameAdam/ensemble_output/kernel/v
?
1Adam/ensemble_output/kernel/v/Read/ReadVariableOpReadVariableOpAdam/ensemble_output/kernel/v*
_output_shapes

:	*
dtype0
?
Adam/ensemble_output/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameAdam/ensemble_output/bias/v
?
/Adam/ensemble_output/bias/v/Read/ReadVariableOpReadVariableOpAdam/ensemble_output/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
??
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*??
value??B?? B??
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
	layer-8

layer_with_weights-5

layer-9
layer_with_weights-6
layer-10
layer-11
layer_with_weights-7
layer-12
layer-13
layer_with_weights-8
layer-14
layer_with_weights-9
layer-15
layer-16
layer_with_weights-10
layer-17
layer_with_weights-11
layer-18
layer_with_weights-12
layer-19
layer-20
layer_with_weights-13
layer-21
layer_with_weights-14
layer-22
layer_with_weights-15
layer-23
layer-24
layer-25
layer_with_weights-16
layer-26
layer_with_weights-17
layer-27
	optimizer
	variables
trainable_variables
 regularization_losses
!	keras_api
"
signatures
 
h

#kernel
$bias
%	variables
&trainable_variables
'regularization_losses
(	keras_api
R
)	variables
*trainable_variables
+regularization_losses
,	keras_api
?
-axis
	.gamma
/beta
0moving_mean
1moving_variance
2	variables
3trainable_variables
4regularization_losses
5	keras_api
h

6kernel
7bias
8	variables
9trainable_variables
:regularization_losses
;	keras_api
R
<	variables
=trainable_variables
>regularization_losses
?	keras_api
?
@axis
	Agamma
Bbeta
Cmoving_mean
Dmoving_variance
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
h

Ikernel
Jbias
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
R
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
?
Saxis
	Tgamma
Ubeta
Vmoving_mean
Wmoving_variance
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
h

\kernel
]bias
^	variables
_trainable_variables
`regularization_losses
a	keras_api
R
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
?
faxis
	ggamma
hbeta
imoving_mean
jmoving_variance
k	variables
ltrainable_variables
mregularization_losses
n	keras_api
 
h

okernel
pbias
q	variables
rtrainable_variables
sregularization_losses
t	keras_api
h

ukernel
vbias
w	variables
xtrainable_variables
yregularization_losses
z	keras_api
R
{	variables
|trainable_variables
}regularization_losses
~	keras_api
m

kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
n
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
n
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
n
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
n
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
n
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
n
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?
	?iter
?beta_1
?beta_2

?decay#m?$m?.m?/m?6m?7m?Am?Bm?Im?Jm?Tm?Um?\m?]m?gm?hm?om?pm?um?vm?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?#v?$v?.v?/v?6v?7v?Av?Bv?Iv?Jv?Tv?Uv?\v?]v?gv?hv?ov?pv?uv?vv?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?
?
#0
$1
.2
/3
04
15
66
77
A8
B9
C10
D11
I12
J13
T14
U15
V16
W17
\18
]19
g20
h21
i22
j23
o24
p25
u26
v27
28
?29
?30
?31
?32
?33
?34
?35
?36
?37
?38
?39
?40
?41
?42
?43
?44
?45
?
#0
$1
.2
/3
64
75
A6
B7
I8
J9
T10
U11
\12
]13
g14
h15
o16
p17
u18
v19
20
?21
?22
?23
?24
?25
?26
?27
?28
?29
?30
?31
?32
?33
?34
?35
 
?
?layer_metrics
	variables
 ?layer_regularization_losses
trainable_variables
?layers
?metrics
 regularization_losses
?non_trainable_variables
 
[Y
VARIABLE_VALUEconv3d_5/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv3d_5/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

#0
$1

#0
$1
 
?
?layer_metrics
%	variables
 ?layer_regularization_losses
&trainable_variables
?layers
?metrics
'regularization_losses
?non_trainable_variables
 
 
 
?
?layer_metrics
)	variables
 ?layer_regularization_losses
*trainable_variables
?layers
?metrics
+regularization_losses
?non_trainable_variables
 
fd
VARIABLE_VALUEbatch_normalization_5/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_5/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_5/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_5/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

.0
/1
02
13

.0
/1
 
?
?layer_metrics
2	variables
 ?layer_regularization_losses
3trainable_variables
?layers
?metrics
4regularization_losses
?non_trainable_variables
[Y
VARIABLE_VALUEconv3d_6/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv3d_6/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

60
71

60
71
 
?
?layer_metrics
8	variables
 ?layer_regularization_losses
9trainable_variables
?layers
?metrics
:regularization_losses
?non_trainable_variables
 
 
 
?
?layer_metrics
<	variables
 ?layer_regularization_losses
=trainable_variables
?layers
?metrics
>regularization_losses
?non_trainable_variables
 
fd
VARIABLE_VALUEbatch_normalization_6/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_6/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_6/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_6/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

A0
B1
C2
D3

A0
B1
 
?
?layer_metrics
E	variables
 ?layer_regularization_losses
Ftrainable_variables
?layers
?metrics
Gregularization_losses
?non_trainable_variables
[Y
VARIABLE_VALUEconv3d_7/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv3d_7/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

I0
J1

I0
J1
 
?
?layer_metrics
K	variables
 ?layer_regularization_losses
Ltrainable_variables
?layers
?metrics
Mregularization_losses
?non_trainable_variables
 
 
 
?
?layer_metrics
O	variables
 ?layer_regularization_losses
Ptrainable_variables
?layers
?metrics
Qregularization_losses
?non_trainable_variables
 
fd
VARIABLE_VALUEbatch_normalization_7/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_7/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_7/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_7/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

T0
U1
V2
W3

T0
U1
 
?
?layer_metrics
X	variables
 ?layer_regularization_losses
Ytrainable_variables
?layers
?metrics
Zregularization_losses
?non_trainable_variables
[Y
VARIABLE_VALUEconv3d_8/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv3d_8/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

\0
]1

\0
]1
 
?
?layer_metrics
^	variables
 ?layer_regularization_losses
_trainable_variables
?layers
?metrics
`regularization_losses
?non_trainable_variables
 
 
 
?
?layer_metrics
b	variables
 ?layer_regularization_losses
ctrainable_variables
?layers
?metrics
dregularization_losses
?non_trainable_variables
 
fd
VARIABLE_VALUEbatch_normalization_8/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_8/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_8/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_8/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

g0
h1
i2
j3

g0
h1
 
?
?layer_metrics
k	variables
 ?layer_regularization_losses
ltrainable_variables
?layers
?metrics
mregularization_losses
?non_trainable_variables
[Y
VARIABLE_VALUEconv3d_9/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv3d_9/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

o0
p1

o0
p1
 
?
?layer_metrics
q	variables
 ?layer_regularization_losses
rtrainable_variables
?layers
?metrics
sregularization_losses
?non_trainable_variables
\Z
VARIABLE_VALUE1_dense18/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUE1_dense18/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE

u0
v1

u0
v1
 
?
?layer_metrics
w	variables
 ?layer_regularization_losses
xtrainable_variables
?layers
?metrics
yregularization_losses
?non_trainable_variables
 
 
 
?
?layer_metrics
{	variables
 ?layer_regularization_losses
|trainable_variables
?layers
?metrics
}regularization_losses
?non_trainable_variables
][
VARIABLE_VALUE2_dense32/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUE2_dense32/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE

0
?1

0
?1
 
?
?layer_metrics
?	variables
 ?layer_regularization_losses
?trainable_variables
?layers
?metrics
?regularization_losses
?non_trainable_variables
 
ge
VARIABLE_VALUEbatch_normalization_9/gamma6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_9/beta5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE!batch_normalization_9/moving_mean<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE%batch_normalization_9/moving_variance@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
?0
?1
?2
?3

?0
?1
 
?
?layer_metrics
?	variables
 ?layer_regularization_losses
?trainable_variables
?layers
?metrics
?regularization_losses
?non_trainable_variables
][
VARIABLE_VALUE3_dense64/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUE3_dense64/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?0
?1
 
?
?layer_metrics
?	variables
 ?layer_regularization_losses
?trainable_variables
?layers
?metrics
?regularization_losses
?non_trainable_variables
 
 
 
?
?layer_metrics
?	variables
 ?layer_regularization_losses
?trainable_variables
?layers
?metrics
?regularization_losses
?non_trainable_variables
][
VARIABLE_VALUE4_dense32/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUE4_dense32/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?0
?1
 
?
?layer_metrics
?	variables
 ?layer_regularization_losses
?trainable_variables
?layers
?metrics
?regularization_losses
?non_trainable_variables
[Y
VARIABLE_VALUEdense_1/kernel7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_1/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?0
?1
 
?
?layer_metrics
?	variables
 ?layer_regularization_losses
?trainable_variables
?layers
?metrics
?regularization_losses
?non_trainable_variables
][
VARIABLE_VALUE5_dense18/kernel7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUE5_dense18/bias5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?0
?1
 
?
?layer_metrics
?	variables
 ?layer_regularization_losses
?trainable_variables
?layers
?metrics
?regularization_losses
?non_trainable_variables
 
 
 
?
?layer_metrics
?	variables
 ?layer_regularization_losses
?trainable_variables
?layers
?metrics
?regularization_losses
?non_trainable_variables
 
 
 
?
?layer_metrics
?	variables
 ?layer_regularization_losses
?trainable_variables
?layers
?metrics
?regularization_losses
?non_trainable_variables
b`
VARIABLE_VALUEcombined_input/kernel7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEcombined_input/bias5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?0
?1
 
?
?layer_metrics
?	variables
 ?layer_regularization_losses
?trainable_variables
?layers
?metrics
?regularization_losses
?non_trainable_variables
ca
VARIABLE_VALUEensemble_output/kernel7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEensemble_output/bias5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?0
?1
 
?
?layer_metrics
?	variables
 ?layer_regularization_losses
?trainable_variables
?layers
?metrics
?regularization_losses
?non_trainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
 
 
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27

?0
?1
H
00
11
C2
D3
V4
W5
i6
j7
?8
?9
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

00
11
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

C0
D1
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

V0
W1
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

i0
j1
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
 
 

?0
?1
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
 
8

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
~|
VARIABLE_VALUEAdam/conv3d_5/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv3d_5/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_5/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/batch_normalization_5/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv3d_6/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv3d_6/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_6/gamma/mQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/batch_normalization_6/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv3d_7/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv3d_7/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_7/gamma/mQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/batch_normalization_7/beta/mPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv3d_8/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv3d_8/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_8/gamma/mQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/batch_normalization_8/beta/mPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv3d_9/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv3d_9/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/1_dense18/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/1_dense18/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/2_dense32/kernel/mSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/2_dense32/bias/mQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_9/gamma/mRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/batch_normalization_9/beta/mQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/3_dense64/kernel/mSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/3_dense64/bias/mQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/4_dense32/kernel/mSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/4_dense32/bias/mQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_1/kernel/mSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_1/bias/mQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/5_dense18/kernel/mSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/5_dense18/bias/mQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/combined_input/kernel/mSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/combined_input/bias/mQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/ensemble_output/kernel/mSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/ensemble_output/bias/mQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv3d_5/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv3d_5/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_5/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/batch_normalization_5/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv3d_6/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv3d_6/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_6/gamma/vQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/batch_normalization_6/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv3d_7/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv3d_7/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_7/gamma/vQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/batch_normalization_7/beta/vPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv3d_8/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv3d_8/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_8/gamma/vQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/batch_normalization_8/beta/vPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv3d_9/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv3d_9/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/1_dense18/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/1_dense18/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/2_dense32/kernel/vSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/2_dense32/bias/vQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_9/gamma/vRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/batch_normalization_9/beta/vQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/3_dense64/kernel/vSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/3_dense64/bias/vQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/4_dense32/kernel/vSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/4_dense32/bias/vQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_1/kernel/vSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_1/bias/vQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/5_dense18/kernel/vSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/5_dense18/bias/vQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/combined_input/kernel/vSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/combined_input/bias/vQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/ensemble_output/kernel/vSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/ensemble_output/bias/vQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_img3d_inputsPlaceholder*5
_output_shapes#
!:???????????*
dtype0**
shape!:???????????
}
serving_default_svm_inputsPlaceholder*'
_output_shapes
:?????????
*
dtype0*
shape:?????????

?
StatefulPartitionedCallStatefulPartitionedCallserving_default_img3d_inputsserving_default_svm_inputsconv3d_5/kernelconv3d_5/biasbatch_normalization_5/gammabatch_normalization_5/beta!batch_normalization_5/moving_mean%batch_normalization_5/moving_varianceconv3d_6/kernelconv3d_6/biasbatch_normalization_6/gammabatch_normalization_6/beta!batch_normalization_6/moving_mean%batch_normalization_6/moving_varianceconv3d_7/kernelconv3d_7/biasbatch_normalization_7/gammabatch_normalization_7/beta!batch_normalization_7/moving_mean%batch_normalization_7/moving_varianceconv3d_8/kernelconv3d_8/biasbatch_normalization_8/gammabatch_normalization_8/beta!batch_normalization_8/moving_mean%batch_normalization_8/moving_varianceconv3d_9/kernelconv3d_9/bias1_dense18/kernel1_dense18/biasbatch_normalization_9/gammabatch_normalization_9/beta!batch_normalization_9/moving_mean%batch_normalization_9/moving_variance2_dense32/kernel2_dense32/bias3_dense64/kernel3_dense64/biasdense_1/kerneldense_1/bias4_dense32/kernel4_dense32/bias5_dense18/kernel5_dense18/biascombined_input/kernelcombined_input/biasensemble_output/kernelensemble_output/bias*;
Tin4
220*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*P
_read_only_resource_inputs2
0.	
 !"#$%&'()*+,-./*0
config_proto 

CPU

GPU2*0J 8? *.
f)R'
%__inference_signature_wrapper_9257026
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?/
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#conv3d_5/kernel/Read/ReadVariableOp!conv3d_5/bias/Read/ReadVariableOp/batch_normalization_5/gamma/Read/ReadVariableOp.batch_normalization_5/beta/Read/ReadVariableOp5batch_normalization_5/moving_mean/Read/ReadVariableOp9batch_normalization_5/moving_variance/Read/ReadVariableOp#conv3d_6/kernel/Read/ReadVariableOp!conv3d_6/bias/Read/ReadVariableOp/batch_normalization_6/gamma/Read/ReadVariableOp.batch_normalization_6/beta/Read/ReadVariableOp5batch_normalization_6/moving_mean/Read/ReadVariableOp9batch_normalization_6/moving_variance/Read/ReadVariableOp#conv3d_7/kernel/Read/ReadVariableOp!conv3d_7/bias/Read/ReadVariableOp/batch_normalization_7/gamma/Read/ReadVariableOp.batch_normalization_7/beta/Read/ReadVariableOp5batch_normalization_7/moving_mean/Read/ReadVariableOp9batch_normalization_7/moving_variance/Read/ReadVariableOp#conv3d_8/kernel/Read/ReadVariableOp!conv3d_8/bias/Read/ReadVariableOp/batch_normalization_8/gamma/Read/ReadVariableOp.batch_normalization_8/beta/Read/ReadVariableOp5batch_normalization_8/moving_mean/Read/ReadVariableOp9batch_normalization_8/moving_variance/Read/ReadVariableOp#conv3d_9/kernel/Read/ReadVariableOp!conv3d_9/bias/Read/ReadVariableOp$1_dense18/kernel/Read/ReadVariableOp"1_dense18/bias/Read/ReadVariableOp$2_dense32/kernel/Read/ReadVariableOp"2_dense32/bias/Read/ReadVariableOp/batch_normalization_9/gamma/Read/ReadVariableOp.batch_normalization_9/beta/Read/ReadVariableOp5batch_normalization_9/moving_mean/Read/ReadVariableOp9batch_normalization_9/moving_variance/Read/ReadVariableOp$3_dense64/kernel/Read/ReadVariableOp"3_dense64/bias/Read/ReadVariableOp$4_dense32/kernel/Read/ReadVariableOp"4_dense32/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp$5_dense18/kernel/Read/ReadVariableOp"5_dense18/bias/Read/ReadVariableOp)combined_input/kernel/Read/ReadVariableOp'combined_input/bias/Read/ReadVariableOp*ensemble_output/kernel/Read/ReadVariableOp(ensemble_output/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp*Adam/conv3d_5/kernel/m/Read/ReadVariableOp(Adam/conv3d_5/bias/m/Read/ReadVariableOp6Adam/batch_normalization_5/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_5/beta/m/Read/ReadVariableOp*Adam/conv3d_6/kernel/m/Read/ReadVariableOp(Adam/conv3d_6/bias/m/Read/ReadVariableOp6Adam/batch_normalization_6/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_6/beta/m/Read/ReadVariableOp*Adam/conv3d_7/kernel/m/Read/ReadVariableOp(Adam/conv3d_7/bias/m/Read/ReadVariableOp6Adam/batch_normalization_7/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_7/beta/m/Read/ReadVariableOp*Adam/conv3d_8/kernel/m/Read/ReadVariableOp(Adam/conv3d_8/bias/m/Read/ReadVariableOp6Adam/batch_normalization_8/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_8/beta/m/Read/ReadVariableOp*Adam/conv3d_9/kernel/m/Read/ReadVariableOp(Adam/conv3d_9/bias/m/Read/ReadVariableOp+Adam/1_dense18/kernel/m/Read/ReadVariableOp)Adam/1_dense18/bias/m/Read/ReadVariableOp+Adam/2_dense32/kernel/m/Read/ReadVariableOp)Adam/2_dense32/bias/m/Read/ReadVariableOp6Adam/batch_normalization_9/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_9/beta/m/Read/ReadVariableOp+Adam/3_dense64/kernel/m/Read/ReadVariableOp)Adam/3_dense64/bias/m/Read/ReadVariableOp+Adam/4_dense32/kernel/m/Read/ReadVariableOp)Adam/4_dense32/bias/m/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp+Adam/5_dense18/kernel/m/Read/ReadVariableOp)Adam/5_dense18/bias/m/Read/ReadVariableOp0Adam/combined_input/kernel/m/Read/ReadVariableOp.Adam/combined_input/bias/m/Read/ReadVariableOp1Adam/ensemble_output/kernel/m/Read/ReadVariableOp/Adam/ensemble_output/bias/m/Read/ReadVariableOp*Adam/conv3d_5/kernel/v/Read/ReadVariableOp(Adam/conv3d_5/bias/v/Read/ReadVariableOp6Adam/batch_normalization_5/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_5/beta/v/Read/ReadVariableOp*Adam/conv3d_6/kernel/v/Read/ReadVariableOp(Adam/conv3d_6/bias/v/Read/ReadVariableOp6Adam/batch_normalization_6/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_6/beta/v/Read/ReadVariableOp*Adam/conv3d_7/kernel/v/Read/ReadVariableOp(Adam/conv3d_7/bias/v/Read/ReadVariableOp6Adam/batch_normalization_7/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_7/beta/v/Read/ReadVariableOp*Adam/conv3d_8/kernel/v/Read/ReadVariableOp(Adam/conv3d_8/bias/v/Read/ReadVariableOp6Adam/batch_normalization_8/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_8/beta/v/Read/ReadVariableOp*Adam/conv3d_9/kernel/v/Read/ReadVariableOp(Adam/conv3d_9/bias/v/Read/ReadVariableOp+Adam/1_dense18/kernel/v/Read/ReadVariableOp)Adam/1_dense18/bias/v/Read/ReadVariableOp+Adam/2_dense32/kernel/v/Read/ReadVariableOp)Adam/2_dense32/bias/v/Read/ReadVariableOp6Adam/batch_normalization_9/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_9/beta/v/Read/ReadVariableOp+Adam/3_dense64/kernel/v/Read/ReadVariableOp)Adam/3_dense64/bias/v/Read/ReadVariableOp+Adam/4_dense32/kernel/v/Read/ReadVariableOp)Adam/4_dense32/bias/v/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOp+Adam/5_dense18/kernel/v/Read/ReadVariableOp)Adam/5_dense18/bias/v/Read/ReadVariableOp0Adam/combined_input/kernel/v/Read/ReadVariableOp.Adam/combined_input/bias/v/Read/ReadVariableOp1Adam/ensemble_output/kernel/v/Read/ReadVariableOp/Adam/ensemble_output/bias/v/Read/ReadVariableOpConst*?
Tin?
?2?	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference__traced_save_9259025
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv3d_5/kernelconv3d_5/biasbatch_normalization_5/gammabatch_normalization_5/beta!batch_normalization_5/moving_mean%batch_normalization_5/moving_varianceconv3d_6/kernelconv3d_6/biasbatch_normalization_6/gammabatch_normalization_6/beta!batch_normalization_6/moving_mean%batch_normalization_6/moving_varianceconv3d_7/kernelconv3d_7/biasbatch_normalization_7/gammabatch_normalization_7/beta!batch_normalization_7/moving_mean%batch_normalization_7/moving_varianceconv3d_8/kernelconv3d_8/biasbatch_normalization_8/gammabatch_normalization_8/beta!batch_normalization_8/moving_mean%batch_normalization_8/moving_varianceconv3d_9/kernelconv3d_9/bias1_dense18/kernel1_dense18/bias2_dense32/kernel2_dense32/biasbatch_normalization_9/gammabatch_normalization_9/beta!batch_normalization_9/moving_mean%batch_normalization_9/moving_variance3_dense64/kernel3_dense64/bias4_dense32/kernel4_dense32/biasdense_1/kerneldense_1/bias5_dense18/kernel5_dense18/biascombined_input/kernelcombined_input/biasensemble_output/kernelensemble_output/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decaytotalcounttotal_1count_1Adam/conv3d_5/kernel/mAdam/conv3d_5/bias/m"Adam/batch_normalization_5/gamma/m!Adam/batch_normalization_5/beta/mAdam/conv3d_6/kernel/mAdam/conv3d_6/bias/m"Adam/batch_normalization_6/gamma/m!Adam/batch_normalization_6/beta/mAdam/conv3d_7/kernel/mAdam/conv3d_7/bias/m"Adam/batch_normalization_7/gamma/m!Adam/batch_normalization_7/beta/mAdam/conv3d_8/kernel/mAdam/conv3d_8/bias/m"Adam/batch_normalization_8/gamma/m!Adam/batch_normalization_8/beta/mAdam/conv3d_9/kernel/mAdam/conv3d_9/bias/mAdam/1_dense18/kernel/mAdam/1_dense18/bias/mAdam/2_dense32/kernel/mAdam/2_dense32/bias/m"Adam/batch_normalization_9/gamma/m!Adam/batch_normalization_9/beta/mAdam/3_dense64/kernel/mAdam/3_dense64/bias/mAdam/4_dense32/kernel/mAdam/4_dense32/bias/mAdam/dense_1/kernel/mAdam/dense_1/bias/mAdam/5_dense18/kernel/mAdam/5_dense18/bias/mAdam/combined_input/kernel/mAdam/combined_input/bias/mAdam/ensemble_output/kernel/mAdam/ensemble_output/bias/mAdam/conv3d_5/kernel/vAdam/conv3d_5/bias/v"Adam/batch_normalization_5/gamma/v!Adam/batch_normalization_5/beta/vAdam/conv3d_6/kernel/vAdam/conv3d_6/bias/v"Adam/batch_normalization_6/gamma/v!Adam/batch_normalization_6/beta/vAdam/conv3d_7/kernel/vAdam/conv3d_7/bias/v"Adam/batch_normalization_7/gamma/v!Adam/batch_normalization_7/beta/vAdam/conv3d_8/kernel/vAdam/conv3d_8/bias/v"Adam/batch_normalization_8/gamma/v!Adam/batch_normalization_8/beta/vAdam/conv3d_9/kernel/vAdam/conv3d_9/bias/vAdam/1_dense18/kernel/vAdam/1_dense18/bias/vAdam/2_dense32/kernel/vAdam/2_dense32/bias/v"Adam/batch_normalization_9/gamma/v!Adam/batch_normalization_9/beta/vAdam/3_dense64/kernel/vAdam/3_dense64/bias/vAdam/4_dense32/kernel/vAdam/4_dense32/bias/vAdam/dense_1/kernel/vAdam/dense_1/bias/vAdam/5_dense18/kernel/vAdam/5_dense18/bias/vAdam/combined_input/kernel/vAdam/combined_input/bias/vAdam/ensemble_output/kernel/vAdam/ensemble_output/bias/v*?
Tin?
?2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference__traced_restore_9259413??
?	
?
7__inference_batch_normalization_7_layer_call_fn_9257975

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8????????????????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_92549552
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*N
_output_shapes<
::8????????????????????????????????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8????????????????????????????????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8????????????????????????????????????@
 
_user_specified_nameinputs
?
s
W__inference_global_average_pooling3d_1_layer_call_and_return_conditional_losses_9255321

inputs
identity?
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2
Mean/reduction_indicesx
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????????????2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:??????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A?????????????????????????????????????????????: {
W
_output_shapesE
C:A?????????????????????????????????????????????
 
_user_specified_nameinputs
?	
?
7__inference_batch_normalization_6_layer_call_fn_9257837

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@@ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_92561742
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:?????????@@ 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????@@ : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:?????????@@ 
 
_user_specified_nameinputs
?
h
L__inference_max_pooling3d_8_layer_call_and_return_conditional_losses_9258108

inputs
identity?
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????*
ksize	
*
paddingVALID*
strides	
2
	MaxPool3D?
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A?????????????????????????????????????????????: {
W
_output_shapesE
C:A?????????????????????????????????????????????
 
_user_specified_nameinputs
?	
?
7__inference_batch_normalization_8_layer_call_fn_9258165

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????@@?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_92560562
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :?????????@@?2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:?????????@@?: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :?????????@@?
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_9257727

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*O
_output_shapes=
;:?????????@@ : : : : :*
data_formatNDHWC*
epsilon%o?:*
is_training( 2
FusedBatchNormV3{
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*3
_output_shapes!
:?????????@@ 2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????@@ : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:[ W
3
_output_shapes!
:?????????@@ 
 
_user_specified_nameinputs
?
?
F__inference_4_dense32_layer_call_and_return_conditional_losses_9255680

inputs0
matmul_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAdd^
	LeakyRelu	LeakyReluBiasAdd:output:0*'
_output_shapes
:????????? 2
	LeakyRelur
IdentityIdentityLeakyRelu:activations:0^NoOp*
T0*'
_output_shapes
:????????? 2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_9258019

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*j
_output_shapesX
V:8????????????????????????????????????@:@:@:@:@:*
data_formatNDHWC*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*N
_output_shapes<
::8????????????????????????????????????@2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8????????????????????????????????????@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:v r
N
_output_shapes<
::8????????????????????????????????????@
 
_user_specified_nameinputs
?
h
L__inference_max_pooling3d_8_layer_call_and_return_conditional_losses_9255024

inputs
identity?
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????*
ksize	
*
paddingVALID*
strides	
2
	MaxPool3D?
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A?????????????????????????????????????????????: {
W
_output_shapesE
C:A?????????????????????????????????????????????
 
_user_specified_nameinputs
?
?
E__inference_conv3d_7_layer_call_and_return_conditional_losses_9255455

inputs<
conv3d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
: @*
dtype02
Conv3D/ReadVariableOp?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@@@*
paddingSAME*
strides	
2
Conv3D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@@@2	
BiasAddj
	LeakyRelu	LeakyReluBiasAdd:output:0*3
_output_shapes!
:?????????@@@2
	LeakyRelu~
IdentityIdentityLeakyRelu:activations:0^NoOp*
T0*3
_output_shapes!
:?????????@@@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@@ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:?????????@@ 
 
_user_specified_nameinputs
?
?
E__inference_conv3d_9_layer_call_and_return_conditional_losses_9255555

inputs>
conv3d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource*,
_output_shapes
:??*
dtype02
Conv3D/ReadVariableOp?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :?????????@@?*
paddingSAME*
strides	
2
Conv3D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :?????????@@?2	
BiasAddk
	LeakyRelu	LeakyReluBiasAdd:output:0*4
_output_shapes"
 :?????????@@?2
	LeakyRelu
IdentityIdentityLeakyRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :?????????@@?2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????@@?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:\ X
4
_output_shapes"
 :?????????@@?
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_9256056

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*T
_output_shapesB
@:?????????@@?:?:?:?:?:*
data_formatNDHWC*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1|
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*4
_output_shapes"
 :?????????@@?2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:?????????@@?: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:\ X
4
_output_shapes"
 :?????????@@?
 
_user_specified_nameinputs
?
M
1__inference_max_pooling3d_8_layer_call_fn_9258098

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:A?????????????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling3d_8_layer_call_and_return_conditional_losses_92550242
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A?????????????????????????????????????????????: {
W
_output_shapesE
C:A?????????????????????????????????????????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_9258237

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*T
_output_shapesB
@:?????????@@?:?:?:?:?:*
data_formatNDHWC*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1|
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*4
_output_shapes"
 :?????????@@?2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:?????????@@?: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:\ X
4
_output_shapes"
 :?????????@@?
 
_user_specified_nameinputs
?	
?
7__inference_batch_normalization_5_layer_call_fn_9257660

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@@ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_92553842
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:?????????@@ 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????@@ : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:?????????@@ 
 
_user_specified_nameinputs
?
h
L__inference_max_pooling3d_5_layer_call_and_return_conditional_losses_9257616

inputs
identity?
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????*
ksize	
*
paddingVALID*
strides	
2
	MaxPool3D?
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A?????????????????????????????????????????????: {
W
_output_shapesE
C:A?????????????????????????????????????????????
 
_user_specified_nameinputs
?
G
+__inference_dropout_1_layer_call_fn_9258548

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_92557082
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_9255207

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*o
_output_shapes]
[:9?????????????????????????????????????:?:?:?:?:*
data_formatNDHWC*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*O
_output_shapes=
;:9?????????????????????????????????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:9?????????????????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:w s
O
_output_shapes=
;:9?????????????????????????????????????
 
_user_specified_nameinputs
Ɋ
?
G__inference_ensemble4d_layer_call_and_return_conditional_losses_9255754

inputs
inputs_1.
conv3d_5_9255356: 
conv3d_5_9255358: +
batch_normalization_5_9255385: +
batch_normalization_5_9255387: +
batch_normalization_5_9255389: +
batch_normalization_5_9255391: .
conv3d_6_9255406:  
conv3d_6_9255408: +
batch_normalization_6_9255435: +
batch_normalization_6_9255437: +
batch_normalization_6_9255439: +
batch_normalization_6_9255441: .
conv3d_7_9255456: @
conv3d_7_9255458:@+
batch_normalization_7_9255485:@+
batch_normalization_7_9255487:@+
batch_normalization_7_9255489:@+
batch_normalization_7_9255491:@/
conv3d_8_9255506:@?
conv3d_8_9255508:	?,
batch_normalization_8_9255535:	?,
batch_normalization_8_9255537:	?,
batch_normalization_8_9255539:	?,
batch_normalization_8_9255541:	?0
conv3d_9_9255556:??
conv3d_9_9255558:	?!
dense18_9255579:

dense18_9255581:,
batch_normalization_9_9255602:	?,
batch_normalization_9_9255604:	?,
batch_normalization_9_9255606:	?,
batch_normalization_9_9255608:	?!
dense32_9255623: 
dense32_9255625: !
dense64_9255647: @
dense64_9255649:@#
dense_1_9255664:
??
dense_1_9255666:	?!
dense32_9255681:@ 
dense32_9255683: !
dense18_9255698: 
dense18_9255700:)
combined_input_9255731:	?	$
combined_input_9255733:	)
ensemble_output_9255748:	%
ensemble_output_9255750:
identity??!1_dense18/StatefulPartitionedCall?!2_dense32/StatefulPartitionedCall?!3_dense64/StatefulPartitionedCall?!4_dense32/StatefulPartitionedCall?!5_dense18/StatefulPartitionedCall?-batch_normalization_5/StatefulPartitionedCall?-batch_normalization_6/StatefulPartitionedCall?-batch_normalization_7/StatefulPartitionedCall?-batch_normalization_8/StatefulPartitionedCall?-batch_normalization_9/StatefulPartitionedCall?&combined_input/StatefulPartitionedCall? conv3d_5/StatefulPartitionedCall? conv3d_6/StatefulPartitionedCall? conv3d_7/StatefulPartitionedCall? conv3d_8/StatefulPartitionedCall? conv3d_9/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?'ensemble_output/StatefulPartitionedCall?
 conv3d_5/StatefulPartitionedCallStatefulPartitionedCallinputs_1conv3d_5_9255356conv3d_5_9255358*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:??????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv3d_5_layer_call_and_return_conditional_losses_92553552"
 conv3d_5/StatefulPartitionedCall?
max_pooling3d_5/PartitionedCallPartitionedCall)conv3d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@@ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling3d_5_layer_call_and_return_conditional_losses_92553652!
max_pooling3d_5/PartitionedCall?
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_5/PartitionedCall:output:0batch_normalization_5_9255385batch_normalization_5_9255387batch_normalization_5_9255389batch_normalization_5_9255391*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@@ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_92553842/
-batch_normalization_5/StatefulPartitionedCall?
 conv3d_6/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0conv3d_6_9255406conv3d_6_9255408*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@@ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv3d_6_layer_call_and_return_conditional_losses_92554052"
 conv3d_6/StatefulPartitionedCall?
max_pooling3d_6/PartitionedCallPartitionedCall)conv3d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@@ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling3d_6_layer_call_and_return_conditional_losses_92554152!
max_pooling3d_6/PartitionedCall?
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_6/PartitionedCall:output:0batch_normalization_6_9255435batch_normalization_6_9255437batch_normalization_6_9255439batch_normalization_6_9255441*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@@ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_92554342/
-batch_normalization_6/StatefulPartitionedCall?
 conv3d_7/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0conv3d_7_9255456conv3d_7_9255458*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv3d_7_layer_call_and_return_conditional_losses_92554552"
 conv3d_7/StatefulPartitionedCall?
max_pooling3d_7/PartitionedCallPartitionedCall)conv3d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling3d_7_layer_call_and_return_conditional_losses_92554652!
max_pooling3d_7/PartitionedCall?
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_7/PartitionedCall:output:0batch_normalization_7_9255485batch_normalization_7_9255487batch_normalization_7_9255489batch_normalization_7_9255491*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@@@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_92554842/
-batch_normalization_7/StatefulPartitionedCall?
 conv3d_8/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0conv3d_8_9255506conv3d_8_9255508*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????@@?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv3d_8_layer_call_and_return_conditional_losses_92555052"
 conv3d_8/StatefulPartitionedCall?
max_pooling3d_8/PartitionedCallPartitionedCall)conv3d_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????@@?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling3d_8_layer_call_and_return_conditional_losses_92555152!
max_pooling3d_8/PartitionedCall?
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_8/PartitionedCall:output:0batch_normalization_8_9255535batch_normalization_8_9255537batch_normalization_8_9255539batch_normalization_8_9255541*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????@@?*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_92555342/
-batch_normalization_8/StatefulPartitionedCall?
 conv3d_9/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0conv3d_9_9255556conv3d_9_9255558*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????@@?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv3d_9_layer_call_and_return_conditional_losses_92555552"
 conv3d_9/StatefulPartitionedCall?
max_pooling3d_9/PartitionedCallPartitionedCall)conv3d_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????@@?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling3d_9_layer_call_and_return_conditional_losses_92555652!
max_pooling3d_9/PartitionedCall?
!1_dense18/StatefulPartitionedCallStatefulPartitionedCallinputsdense18_9255579dense18_9255581*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_1_dense18_layer_call_and_return_conditional_losses_92555782#
!1_dense18/StatefulPartitionedCall?
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_9/PartitionedCall:output:0batch_normalization_9_9255602batch_normalization_9_9255604batch_normalization_9_9255606batch_normalization_9_9255608*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????@@?*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_92556012/
-batch_normalization_9/StatefulPartitionedCall?
!2_dense32/StatefulPartitionedCallStatefulPartitionedCall*1_dense18/StatefulPartitionedCall:output:0dense32_9255623dense32_9255625*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_2_dense32_layer_call_and_return_conditional_losses_92556222#
!2_dense32/StatefulPartitionedCall?
*global_average_pooling3d_1/PartitionedCallPartitionedCall6batch_normalization_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *`
f[RY
W__inference_global_average_pooling3d_1_layer_call_and_return_conditional_losses_92556332,
*global_average_pooling3d_1/PartitionedCall?
!3_dense64/StatefulPartitionedCallStatefulPartitionedCall*2_dense32/StatefulPartitionedCall:output:0dense64_9255647dense64_9255649*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_3_dense64_layer_call_and_return_conditional_losses_92556462#
!3_dense64/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling3d_1/PartitionedCall:output:0dense_1_9255664dense_1_9255666*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_92556632!
dense_1/StatefulPartitionedCall?
!4_dense32/StatefulPartitionedCallStatefulPartitionedCall*3_dense64/StatefulPartitionedCall:output:0dense32_9255681dense32_9255683*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_4_dense32_layer_call_and_return_conditional_losses_92556802#
!4_dense32/StatefulPartitionedCall?
!5_dense18/StatefulPartitionedCallStatefulPartitionedCall*4_dense32/StatefulPartitionedCall:output:0dense18_9255698dense18_9255700*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_5_dense18_layer_call_and_return_conditional_losses_92556972#
!5_dense18/StatefulPartitionedCall?
dropout_1/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_92557082
dropout_1/PartitionedCall?
concatenate_1/PartitionedCallPartitionedCall*5_dense18/StatefulPartitionedCall:output:0"dropout_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_concatenate_1_layer_call_and_return_conditional_losses_92557172
concatenate_1/PartitionedCall?
&combined_input/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0combined_input_9255731combined_input_9255733*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_combined_input_layer_call_and_return_conditional_losses_92557302(
&combined_input/StatefulPartitionedCall?
'ensemble_output/StatefulPartitionedCallStatefulPartitionedCall/combined_input/StatefulPartitionedCall:output:0ensemble_output_9255748ensemble_output_9255750*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_ensemble_output_layer_call_and_return_conditional_losses_92557472)
'ensemble_output/StatefulPartitionedCall?
IdentityIdentity0ensemble_output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp"^1_dense18/StatefulPartitionedCall"^2_dense32/StatefulPartitionedCall"^3_dense64/StatefulPartitionedCall"^4_dense32/StatefulPartitionedCall"^5_dense18/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall.^batch_normalization_9/StatefulPartitionedCall'^combined_input/StatefulPartitionedCall!^conv3d_5/StatefulPartitionedCall!^conv3d_6/StatefulPartitionedCall!^conv3d_7/StatefulPartitionedCall!^conv3d_8/StatefulPartitionedCall!^conv3d_9/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall(^ensemble_output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????
:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2F
!1_dense18/StatefulPartitionedCall!1_dense18/StatefulPartitionedCall2F
!2_dense32/StatefulPartitionedCall!2_dense32/StatefulPartitionedCall2F
!3_dense64/StatefulPartitionedCall!3_dense64/StatefulPartitionedCall2F
!4_dense32/StatefulPartitionedCall!4_dense32/StatefulPartitionedCall2F
!5_dense18/StatefulPartitionedCall!5_dense18/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2P
&combined_input/StatefulPartitionedCall&combined_input/StatefulPartitionedCall2D
 conv3d_5/StatefulPartitionedCall conv3d_5/StatefulPartitionedCall2D
 conv3d_6/StatefulPartitionedCall conv3d_6/StatefulPartitionedCall2D
 conv3d_7/StatefulPartitionedCall conv3d_7/StatefulPartitionedCall2D
 conv3d_8/StatefulPartitionedCall conv3d_8/StatefulPartitionedCall2D
 conv3d_9/StatefulPartitionedCall conv3d_9/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2R
'ensemble_output/StatefulPartitionedCall'ensemble_output/StatefulPartitionedCall:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs:]Y
5
_output_shapes#
!:???????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_9254659

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*j
_output_shapesX
V:8???????????????????????????????????? : : : : :*
data_formatNDHWC*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*N
_output_shapes<
::8???????????????????????????????????? 2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8???????????????????????????????????? : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:v r
N
_output_shapes<
::8???????????????????????????????????? 
 
_user_specified_nameinputs
?
?
F__inference_3_dense64_layer_call_and_return_conditional_losses_9258461

inputs0
matmul_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: @*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAdd^
	LeakyRelu	LeakyReluBiasAdd:output:0*'
_output_shapes
:?????????@2
	LeakyRelur
IdentityIdentityLeakyRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
K__inference_combined_input_layer_call_and_return_conditional_losses_9255730

inputs1
matmul_readvariableop_resource:	?	-
biasadd_readvariableop_resource:	
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2	
BiasAdd^
	LeakyRelu	LeakyReluBiasAdd:output:0*'
_output_shapes
:?????????	2
	LeakyRelur
IdentityIdentityLeakyRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????	2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
F__inference_3_dense64_layer_call_and_return_conditional_losses_9255646

inputs0
matmul_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: @*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAdd^
	LeakyRelu	LeakyReluBiasAdd:output:0*'
_output_shapes
:?????????@2
	LeakyRelur
IdentityIdentityLeakyRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_9256233

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*O
_output_shapes=
;:?????????@@ : : : : :*
data_formatNDHWC*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1{
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*3
_output_shapes!
:?????????@@ 2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????@@ : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:[ W
3
_output_shapes!
:?????????@@ 
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_9257873

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*j
_output_shapesX
V:8???????????????????????????????????? : : : : :*
data_formatNDHWC*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*N
_output_shapes<
::8???????????????????????????????????? 2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8???????????????????????????????????? : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:v r
N
_output_shapes<
::8???????????????????????????????????? 
 
_user_specified_nameinputs
?
?
F__inference_2_dense32_layer_call_and_return_conditional_losses_9255622

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAdd^
	LeakyRelu	LeakyReluBiasAdd:output:0*'
_output_shapes
:????????? 2
	LeakyRelur
IdentityIdentityLeakyRelu:activations:0^NoOp*
T0*'
_output_shapes
:????????? 2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
[
/__inference_concatenate_1_layer_call_fn_9258576
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_concatenate_1_layer_call_and_return_conditional_losses_92557172
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????:??????????:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/1
?
?
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_9258219

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*T
_output_shapesB
@:?????????@@?:?:?:?:?:*
data_formatNDHWC*
epsilon%o?:*
is_training( 2
FusedBatchNormV3|
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*4
_output_shapes"
 :?????????@@?2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:?????????@@?: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:\ X
4
_output_shapes"
 :?????????@@?
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_9255601

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*T
_output_shapesB
@:?????????@@?:?:?:?:?:*
data_formatNDHWC*
epsilon%o?:*
is_training( 2
FusedBatchNormV3|
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*4
_output_shapes"
 :?????????@@?2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:?????????@@?: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:\ X
4
_output_shapes"
 :?????????@@?
 
_user_specified_nameinputs
?
h
L__inference_max_pooling3d_9_layer_call_and_return_conditional_losses_9255565

inputs
identity?
	MaxPool3D	MaxPool3Dinputs*
T0*4
_output_shapes"
 :?????????@@?*
ksize	
*
paddingVALID*
strides	
2
	MaxPool3Ds
IdentityIdentityMaxPool3D:output:0*
T0*4
_output_shapes"
 :?????????@@?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????@@?:\ X
4
_output_shapes"
 :?????????@@?
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_9255251

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*o
_output_shapes]
[:9?????????????????????????????????????:?:?:?:?:*
data_formatNDHWC*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*O
_output_shapes=
;:9?????????????????????????????????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:9?????????????????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:w s
O
_output_shapes=
;:9?????????????????????????????????????
 
_user_specified_nameinputs
?
?
F__inference_5_dense18_layer_call_and_return_conditional_losses_9255697

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd^
	LeakyRelu	LeakyReluBiasAdd:output:0*'
_output_shapes
:?????????2
	LeakyRelur
IdentityIdentityLeakyRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
??
?
G__inference_ensemble4d_layer_call_and_return_conditional_losses_9256483

inputs
inputs_1.
conv3d_5_9256364: 
conv3d_5_9256366: +
batch_normalization_5_9256370: +
batch_normalization_5_9256372: +
batch_normalization_5_9256374: +
batch_normalization_5_9256376: .
conv3d_6_9256379:  
conv3d_6_9256381: +
batch_normalization_6_9256385: +
batch_normalization_6_9256387: +
batch_normalization_6_9256389: +
batch_normalization_6_9256391: .
conv3d_7_9256394: @
conv3d_7_9256396:@+
batch_normalization_7_9256400:@+
batch_normalization_7_9256402:@+
batch_normalization_7_9256404:@+
batch_normalization_7_9256406:@/
conv3d_8_9256409:@?
conv3d_8_9256411:	?,
batch_normalization_8_9256415:	?,
batch_normalization_8_9256417:	?,
batch_normalization_8_9256419:	?,
batch_normalization_8_9256421:	?0
conv3d_9_9256424:??
conv3d_9_9256426:	?!
dense18_9256430:

dense18_9256432:,
batch_normalization_9_9256435:	?,
batch_normalization_9_9256437:	?,
batch_normalization_9_9256439:	?,
batch_normalization_9_9256441:	?!
dense32_9256444: 
dense32_9256446: !
dense64_9256450: @
dense64_9256452:@#
dense_1_9256455:
??
dense_1_9256457:	?!
dense32_9256460:@ 
dense32_9256462: !
dense18_9256465: 
dense18_9256467:)
combined_input_9256472:	?	$
combined_input_9256474:	)
ensemble_output_9256477:	%
ensemble_output_9256479:
identity??!1_dense18/StatefulPartitionedCall?!2_dense32/StatefulPartitionedCall?!3_dense64/StatefulPartitionedCall?!4_dense32/StatefulPartitionedCall?!5_dense18/StatefulPartitionedCall?-batch_normalization_5/StatefulPartitionedCall?-batch_normalization_6/StatefulPartitionedCall?-batch_normalization_7/StatefulPartitionedCall?-batch_normalization_8/StatefulPartitionedCall?-batch_normalization_9/StatefulPartitionedCall?&combined_input/StatefulPartitionedCall? conv3d_5/StatefulPartitionedCall? conv3d_6/StatefulPartitionedCall? conv3d_7/StatefulPartitionedCall? conv3d_8/StatefulPartitionedCall? conv3d_9/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?'ensemble_output/StatefulPartitionedCall?
 conv3d_5/StatefulPartitionedCallStatefulPartitionedCallinputs_1conv3d_5_9256364conv3d_5_9256366*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:??????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv3d_5_layer_call_and_return_conditional_losses_92553552"
 conv3d_5/StatefulPartitionedCall?
max_pooling3d_5/PartitionedCallPartitionedCall)conv3d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@@ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling3d_5_layer_call_and_return_conditional_losses_92553652!
max_pooling3d_5/PartitionedCall?
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_5/PartitionedCall:output:0batch_normalization_5_9256370batch_normalization_5_9256372batch_normalization_5_9256374batch_normalization_5_9256376*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@@ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_92562332/
-batch_normalization_5/StatefulPartitionedCall?
 conv3d_6/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0conv3d_6_9256379conv3d_6_9256381*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@@ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv3d_6_layer_call_and_return_conditional_losses_92554052"
 conv3d_6/StatefulPartitionedCall?
max_pooling3d_6/PartitionedCallPartitionedCall)conv3d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@@ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling3d_6_layer_call_and_return_conditional_losses_92554152!
max_pooling3d_6/PartitionedCall?
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_6/PartitionedCall:output:0batch_normalization_6_9256385batch_normalization_6_9256387batch_normalization_6_9256389batch_normalization_6_9256391*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@@ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_92561742/
-batch_normalization_6/StatefulPartitionedCall?
 conv3d_7/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0conv3d_7_9256394conv3d_7_9256396*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv3d_7_layer_call_and_return_conditional_losses_92554552"
 conv3d_7/StatefulPartitionedCall?
max_pooling3d_7/PartitionedCallPartitionedCall)conv3d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling3d_7_layer_call_and_return_conditional_losses_92554652!
max_pooling3d_7/PartitionedCall?
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_7/PartitionedCall:output:0batch_normalization_7_9256400batch_normalization_7_9256402batch_normalization_7_9256404batch_normalization_7_9256406*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_92561152/
-batch_normalization_7/StatefulPartitionedCall?
 conv3d_8/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0conv3d_8_9256409conv3d_8_9256411*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????@@?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv3d_8_layer_call_and_return_conditional_losses_92555052"
 conv3d_8/StatefulPartitionedCall?
max_pooling3d_8/PartitionedCallPartitionedCall)conv3d_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????@@?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling3d_8_layer_call_and_return_conditional_losses_92555152!
max_pooling3d_8/PartitionedCall?
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_8/PartitionedCall:output:0batch_normalization_8_9256415batch_normalization_8_9256417batch_normalization_8_9256419batch_normalization_8_9256421*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????@@?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_92560562/
-batch_normalization_8/StatefulPartitionedCall?
 conv3d_9/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0conv3d_9_9256424conv3d_9_9256426*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????@@?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv3d_9_layer_call_and_return_conditional_losses_92555552"
 conv3d_9/StatefulPartitionedCall?
max_pooling3d_9/PartitionedCallPartitionedCall)conv3d_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????@@?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling3d_9_layer_call_and_return_conditional_losses_92555652!
max_pooling3d_9/PartitionedCall?
!1_dense18/StatefulPartitionedCallStatefulPartitionedCallinputsdense18_9256430dense18_9256432*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_1_dense18_layer_call_and_return_conditional_losses_92555782#
!1_dense18/StatefulPartitionedCall?
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_9/PartitionedCall:output:0batch_normalization_9_9256435batch_normalization_9_9256437batch_normalization_9_9256439batch_normalization_9_9256441*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????@@?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_92559872/
-batch_normalization_9/StatefulPartitionedCall?
!2_dense32/StatefulPartitionedCallStatefulPartitionedCall*1_dense18/StatefulPartitionedCall:output:0dense32_9256444dense32_9256446*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_2_dense32_layer_call_and_return_conditional_losses_92556222#
!2_dense32/StatefulPartitionedCall?
*global_average_pooling3d_1/PartitionedCallPartitionedCall6batch_normalization_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *`
f[RY
W__inference_global_average_pooling3d_1_layer_call_and_return_conditional_losses_92556332,
*global_average_pooling3d_1/PartitionedCall?
!3_dense64/StatefulPartitionedCallStatefulPartitionedCall*2_dense32/StatefulPartitionedCall:output:0dense64_9256450dense64_9256452*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_3_dense64_layer_call_and_return_conditional_losses_92556462#
!3_dense64/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling3d_1/PartitionedCall:output:0dense_1_9256455dense_1_9256457*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_92556632!
dense_1/StatefulPartitionedCall?
!4_dense32/StatefulPartitionedCallStatefulPartitionedCall*3_dense64/StatefulPartitionedCall:output:0dense32_9256460dense32_9256462*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_4_dense32_layer_call_and_return_conditional_losses_92556802#
!4_dense32/StatefulPartitionedCall?
!5_dense18/StatefulPartitionedCallStatefulPartitionedCall*4_dense32/StatefulPartitionedCall:output:0dense18_9256465dense18_9256467*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_5_dense18_layer_call_and_return_conditional_losses_92556972#
!5_dense18/StatefulPartitionedCall?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_92558962#
!dropout_1/StatefulPartitionedCall?
concatenate_1/PartitionedCallPartitionedCall*5_dense18/StatefulPartitionedCall:output:0*dropout_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_concatenate_1_layer_call_and_return_conditional_losses_92557172
concatenate_1/PartitionedCall?
&combined_input/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0combined_input_9256472combined_input_9256474*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_combined_input_layer_call_and_return_conditional_losses_92557302(
&combined_input/StatefulPartitionedCall?
'ensemble_output/StatefulPartitionedCallStatefulPartitionedCall/combined_input/StatefulPartitionedCall:output:0ensemble_output_9256477ensemble_output_9256479*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_ensemble_output_layer_call_and_return_conditional_losses_92557472)
'ensemble_output/StatefulPartitionedCall?
IdentityIdentity0ensemble_output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp"^1_dense18/StatefulPartitionedCall"^2_dense32/StatefulPartitionedCall"^3_dense64/StatefulPartitionedCall"^4_dense32/StatefulPartitionedCall"^5_dense18/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall.^batch_normalization_9/StatefulPartitionedCall'^combined_input/StatefulPartitionedCall!^conv3d_5/StatefulPartitionedCall!^conv3d_6/StatefulPartitionedCall!^conv3d_7/StatefulPartitionedCall!^conv3d_8/StatefulPartitionedCall!^conv3d_9/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall(^ensemble_output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????
:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2F
!1_dense18/StatefulPartitionedCall!1_dense18/StatefulPartitionedCall2F
!2_dense32/StatefulPartitionedCall!2_dense32/StatefulPartitionedCall2F
!3_dense64/StatefulPartitionedCall!3_dense64/StatefulPartitionedCall2F
!4_dense32/StatefulPartitionedCall!4_dense32/StatefulPartitionedCall2F
!5_dense18/StatefulPartitionedCall!5_dense18/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2P
&combined_input/StatefulPartitionedCall&combined_input/StatefulPartitionedCall2D
 conv3d_5/StatefulPartitionedCall conv3d_5/StatefulPartitionedCall2D
 conv3d_6/StatefulPartitionedCall conv3d_6/StatefulPartitionedCall2D
 conv3d_7/StatefulPartitionedCall conv3d_7/StatefulPartitionedCall2D
 conv3d_8/StatefulPartitionedCall conv3d_8/StatefulPartitionedCall2D
 conv3d_9/StatefulPartitionedCall conv3d_9/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2R
'ensemble_output/StatefulPartitionedCall'ensemble_output/StatefulPartitionedCall:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs:]Y
5
_output_shapes#
!:???????????
 
_user_specified_nameinputs
?
?
E__inference_conv3d_8_layer_call_and_return_conditional_losses_9258093

inputs=
conv3d_readvariableop_resource:@?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource*+
_output_shapes
:@?*
dtype02
Conv3D/ReadVariableOp?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :?????????@@?*
paddingSAME*
strides	
2
Conv3D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :?????????@@?2	
BiasAddk
	LeakyRelu	LeakyReluBiasAdd:output:0*4
_output_shapes"
 :?????????@@?2
	LeakyRelu
IdentityIdentityLeakyRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :?????????@@?2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:?????????@@@
 
_user_specified_nameinputs
?
?
*__inference_conv3d_8_layer_call_fn_9258082

inputs&
unknown:@?
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????@@?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv3d_8_layer_call_and_return_conditional_losses_92555052
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :?????????@@?2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@@@: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:?????????@@@
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_9254615

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*j
_output_shapesX
V:8???????????????????????????????????? : : : : :*
data_formatNDHWC*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*N
_output_shapes<
::8???????????????????????????????????? 2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8???????????????????????????????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:v r
N
_output_shapes<
::8???????????????????????????????????? 
 
_user_specified_nameinputs
?
?
E__inference_conv3d_9_layer_call_and_return_conditional_losses_9258257

inputs>
conv3d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource*,
_output_shapes
:??*
dtype02
Conv3D/ReadVariableOp?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :?????????@@?*
paddingSAME*
strides	
2
Conv3D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :?????????@@?2	
BiasAddk
	LeakyRelu	LeakyReluBiasAdd:output:0*4
_output_shapes"
 :?????????@@?2
	LeakyRelu
IdentityIdentityLeakyRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :?????????@@?2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????@@?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:\ X
4
_output_shapes"
 :?????????@@?
 
_user_specified_nameinputs
?
?
F__inference_4_dense32_layer_call_and_return_conditional_losses_9258503

inputs0
matmul_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAdd^
	LeakyRelu	LeakyReluBiasAdd:output:0*'
_output_shapes
:????????? 2
	LeakyRelur
IdentityIdentityLeakyRelu:activations:0^NoOp*
T0*'
_output_shapes
:????????? 2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
h
L__inference_max_pooling3d_9_layer_call_and_return_conditional_losses_9255172

inputs
identity?
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????*
ksize	
*
paddingVALID*
strides	
2
	MaxPool3D?
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A?????????????????????????????????????????????: {
W
_output_shapesE
C:A?????????????????????????????????????????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_9258405

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*o
_output_shapes]
[:9?????????????????????????????????????:?:?:?:?:*
data_formatNDHWC*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*O
_output_shapes=
;:9?????????????????????????????????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:9?????????????????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:w s
O
_output_shapes=
;:9?????????????????????????????????????
 
_user_specified_nameinputs
?
h
L__inference_max_pooling3d_7_layer_call_and_return_conditional_losses_9257949

inputs
identity?
	MaxPool3D	MaxPool3Dinputs*
T0*3
_output_shapes!
:?????????@@@*
ksize	
*
paddingVALID*
strides	
2
	MaxPool3Dr
IdentityIdentityMaxPool3D:output:0*
T0*3
_output_shapes!
:?????????@@@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@@:[ W
3
_output_shapes!
:?????????@@@
 
_user_specified_nameinputs
?
?
,__inference_ensemble4d_layer_call_fn_9257124
inputs_0
inputs_1%
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: '
	unknown_5:  
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: (

unknown_11: @

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@)

unknown_17:@?

unknown_18:	?

unknown_19:	?

unknown_20:	?

unknown_21:	?

unknown_22:	?*

unknown_23:??

unknown_24:	?

unknown_25:


unknown_26:

unknown_27:	?

unknown_28:	?

unknown_29:	?

unknown_30:	?

unknown_31: 

unknown_32: 

unknown_33: @

unknown_34:@

unknown_35:
??

unknown_36:	?

unknown_37:@ 

unknown_38: 

unknown_39: 

unknown_40:

unknown_41:	?	

unknown_42:	

unknown_43:	

unknown_44:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44*;
Tin4
220*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*P
_read_only_resource_inputs2
0.	
 !"#$%&'()*+,-./*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_ensemble4d_layer_call_and_return_conditional_losses_92557542
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????
:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????

"
_user_specified_name
inputs/0:_[
5
_output_shapes#
!:???????????
"
_user_specified_name
inputs/1
?
?
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_9257691

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*j
_output_shapesX
V:8???????????????????????????????????? : : : : :*
data_formatNDHWC*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*N
_output_shapes<
::8???????????????????????????????????? 2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8???????????????????????????????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:v r
N
_output_shapes<
::8???????????????????????????????????? 
 
_user_specified_nameinputs
?	
?
7__inference_batch_normalization_8_layer_call_fn_9258139

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *O
_output_shapes=
;:9?????????????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_92551032
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*O
_output_shapes=
;:9?????????????????????????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:9?????????????????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:w s
O
_output_shapes=
;:9?????????????????????????????????????
 
_user_specified_nameinputs
?
s
W__inference_global_average_pooling3d_1_layer_call_and_return_conditional_losses_9258477

inputs
identity?
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2
Mean/reduction_indicesx
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????????????2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:??????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A?????????????????????????????????????????????: {
W
_output_shapesE
C:A?????????????????????????????????????????????
 
_user_specified_nameinputs
?
d
F__inference_dropout_1_layer_call_and_return_conditional_losses_9258558

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_9255987

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*T
_output_shapesB
@:?????????@@?:?:?:?:?:*
data_formatNDHWC*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1|
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*4
_output_shapes"
 :?????????@@?2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:?????????@@?: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:\ X
4
_output_shapes"
 :?????????@@?
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_9258037

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*j
_output_shapesX
V:8????????????????????????????????????@:@:@:@:@:*
data_formatNDHWC*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*N
_output_shapes<
::8????????????????????????????????????@2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8????????????????????????????????????@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:v r
N
_output_shapes<
::8????????????????????????????????????@
 
_user_specified_nameinputs
?
d
+__inference_dropout_1_layer_call_fn_9258553

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_92558962
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_9258441

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*T
_output_shapesB
@:?????????@@?:?:?:?:?:*
data_formatNDHWC*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1|
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*4
_output_shapes"
 :?????????@@?2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:?????????@@?: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:\ X
4
_output_shapes"
 :?????????@@?
 
_user_specified_nameinputs
?
h
L__inference_max_pooling3d_6_layer_call_and_return_conditional_losses_9257780

inputs
identity?
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????*
ksize	
*
paddingVALID*
strides	
2
	MaxPool3D?
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A?????????????????????????????????????????????: {
W
_output_shapesE
C:A?????????????????????????????????????????????
 
_user_specified_nameinputs
?
?
E__inference_conv3d_5_layer_call_and_return_conditional_losses_9257601

inputs<
conv3d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
: *
dtype02
Conv3D/ReadVariableOp?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:??????????? *
paddingSAME*
strides	
2
Conv3D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:??????????? 2	
BiasAddl
	LeakyRelu	LeakyReluBiasAdd:output:0*5
_output_shapes#
!:??????????? 2
	LeakyRelu?
IdentityIdentityLeakyRelu:activations:0^NoOp*
T0*5
_output_shapes#
!:??????????? 2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:] Y
5
_output_shapes#
!:???????????
 
_user_specified_nameinputs
?
M
1__inference_max_pooling3d_6_layer_call_fn_9257770

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:A?????????????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling3d_6_layer_call_and_return_conditional_losses_92547282
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A?????????????????????????????????????????????: {
W
_output_shapesE
C:A?????????????????????????????????????????????
 
_user_specified_nameinputs
?
h
L__inference_max_pooling3d_8_layer_call_and_return_conditional_losses_9255515

inputs
identity?
	MaxPool3D	MaxPool3Dinputs*
T0*4
_output_shapes"
 :?????????@@?*
ksize	
*
paddingVALID*
strides	
2
	MaxPool3Ds
IdentityIdentityMaxPool3D:output:0*
T0*4
_output_shapes"
 :?????????@@?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????@@?:\ X
4
_output_shapes"
 :?????????@@?
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_9255484

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*O
_output_shapes=
;:?????????@@@:@:@:@:@:*
data_formatNDHWC*
epsilon%o?:*
is_training( 2
FusedBatchNormV3{
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*3
_output_shapes!
:?????????@@@2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????@@@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:[ W
3
_output_shapes!
:?????????@@@
 
_user_specified_nameinputs
?	
?
7__inference_batch_normalization_5_layer_call_fn_9257634

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8???????????????????????????????????? *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_92546152
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*N
_output_shapes<
::8???????????????????????????????????? 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8???????????????????????????????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8???????????????????????????????????? 
 
_user_specified_nameinputs
?
?
E__inference_conv3d_8_layer_call_and_return_conditional_losses_9255505

inputs=
conv3d_readvariableop_resource:@?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource*+
_output_shapes
:@?*
dtype02
Conv3D/ReadVariableOp?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :?????????@@?*
paddingSAME*
strides	
2
Conv3D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :?????????@@?2	
BiasAddk
	LeakyRelu	LeakyReluBiasAdd:output:0*4
_output_shapes"
 :?????????@@?2
	LeakyRelu
IdentityIdentityLeakyRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :?????????@@?2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:?????????@@@
 
_user_specified_nameinputs
?
h
L__inference_max_pooling3d_5_layer_call_and_return_conditional_losses_9255365

inputs
identity?
	MaxPool3D	MaxPool3Dinputs*
T0*3
_output_shapes!
:?????????@@ *
ksize	
*
paddingVALID*
strides	
2
	MaxPool3Dr
IdentityIdentityMaxPool3D:output:0*
T0*3
_output_shapes!
:?????????@@ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:??????????? :] Y
5
_output_shapes#
!:??????????? 
 
_user_specified_nameinputs
?	
?
7__inference_batch_normalization_7_layer_call_fn_9257988

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@@@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_92554842
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:?????????@@@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????@@@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:?????????@@@
 
_user_specified_nameinputs
?
?
E__inference_conv3d_6_layer_call_and_return_conditional_losses_9257765

inputs<
conv3d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:  *
dtype02
Conv3D/ReadVariableOp?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@@ *
paddingSAME*
strides	
2
Conv3D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@@ 2	
BiasAddj
	LeakyRelu	LeakyReluBiasAdd:output:0*3
_output_shapes!
:?????????@@ 2
	LeakyRelu~
IdentityIdentityLeakyRelu:activations:0^NoOp*
T0*3
_output_shapes!
:?????????@@ 2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@@ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:?????????@@ 
 
_user_specified_nameinputs
?
X
<__inference_global_average_pooling3d_1_layer_call_fn_9258471

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *`
f[RY
W__inference_global_average_pooling3d_1_layer_call_and_return_conditional_losses_92556332
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????@@?:\ X
4
_output_shapes"
 :?????????@@?
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_9258201

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*o
_output_shapes]
[:9?????????????????????????????????????:?:?:?:?:*
data_formatNDHWC*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*O
_output_shapes=
;:9?????????????????????????????????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:9?????????????????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:w s
O
_output_shapes=
;:9?????????????????????????????????????
 
_user_specified_nameinputs
?	
?
7__inference_batch_normalization_9_layer_call_fn_9258369

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????@@?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_92559872
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :?????????@@?2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:?????????@@?: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :?????????@@?
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_9255434

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*O
_output_shapes=
;:?????????@@ : : : : :*
data_formatNDHWC*
epsilon%o?:*
is_training( 2
FusedBatchNormV3{
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*3
_output_shapes!
:?????????@@ 2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????@@ : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:[ W
3
_output_shapes!
:?????????@@ 
 
_user_specified_nameinputs
?	
?
7__inference_batch_normalization_6_layer_call_fn_9257798

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8???????????????????????????????????? *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_92547632
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*N
_output_shapes<
::8???????????????????????????????????? 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8???????????????????????????????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8???????????????????????????????????? 
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_9257909

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*O
_output_shapes=
;:?????????@@ : : : : :*
data_formatNDHWC*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1{
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*3
_output_shapes!
:?????????@@ 2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????@@ : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:[ W
3
_output_shapes!
:?????????@@ 
 
_user_specified_nameinputs
?
?
E__inference_conv3d_5_layer_call_and_return_conditional_losses_9255355

inputs<
conv3d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
: *
dtype02
Conv3D/ReadVariableOp?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:??????????? *
paddingSAME*
strides	
2
Conv3D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:??????????? 2	
BiasAddl
	LeakyRelu	LeakyReluBiasAdd:output:0*5
_output_shapes#
!:??????????? 2
	LeakyRelu?
IdentityIdentityLeakyRelu:activations:0^NoOp*
T0*5
_output_shapes#
!:??????????? 2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:] Y
5
_output_shapes#
!:???????????
 
_user_specified_nameinputs
?	
?
7__inference_batch_normalization_9_layer_call_fn_9258343

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *O
_output_shapes=
;:9?????????????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_92552512
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*O
_output_shapes=
;:9?????????????????????????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:9?????????????????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:w s
O
_output_shapes=
;:9?????????????????????????????????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_9255059

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*o
_output_shapes]
[:9?????????????????????????????????????:?:?:?:?:*
data_formatNDHWC*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*O
_output_shapes=
;:9?????????????????????????????????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:9?????????????????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:w s
O
_output_shapes=
;:9?????????????????????????????????????
 
_user_specified_nameinputs
??
?8
 __inference__traced_save_9259025
file_prefix.
*savev2_conv3d_5_kernel_read_readvariableop,
(savev2_conv3d_5_bias_read_readvariableop:
6savev2_batch_normalization_5_gamma_read_readvariableop9
5savev2_batch_normalization_5_beta_read_readvariableop@
<savev2_batch_normalization_5_moving_mean_read_readvariableopD
@savev2_batch_normalization_5_moving_variance_read_readvariableop.
*savev2_conv3d_6_kernel_read_readvariableop,
(savev2_conv3d_6_bias_read_readvariableop:
6savev2_batch_normalization_6_gamma_read_readvariableop9
5savev2_batch_normalization_6_beta_read_readvariableop@
<savev2_batch_normalization_6_moving_mean_read_readvariableopD
@savev2_batch_normalization_6_moving_variance_read_readvariableop.
*savev2_conv3d_7_kernel_read_readvariableop,
(savev2_conv3d_7_bias_read_readvariableop:
6savev2_batch_normalization_7_gamma_read_readvariableop9
5savev2_batch_normalization_7_beta_read_readvariableop@
<savev2_batch_normalization_7_moving_mean_read_readvariableopD
@savev2_batch_normalization_7_moving_variance_read_readvariableop.
*savev2_conv3d_8_kernel_read_readvariableop,
(savev2_conv3d_8_bias_read_readvariableop:
6savev2_batch_normalization_8_gamma_read_readvariableop9
5savev2_batch_normalization_8_beta_read_readvariableop@
<savev2_batch_normalization_8_moving_mean_read_readvariableopD
@savev2_batch_normalization_8_moving_variance_read_readvariableop.
*savev2_conv3d_9_kernel_read_readvariableop,
(savev2_conv3d_9_bias_read_readvariableop/
+savev2_1_dense18_kernel_read_readvariableop-
)savev2_1_dense18_bias_read_readvariableop/
+savev2_2_dense32_kernel_read_readvariableop-
)savev2_2_dense32_bias_read_readvariableop:
6savev2_batch_normalization_9_gamma_read_readvariableop9
5savev2_batch_normalization_9_beta_read_readvariableop@
<savev2_batch_normalization_9_moving_mean_read_readvariableopD
@savev2_batch_normalization_9_moving_variance_read_readvariableop/
+savev2_3_dense64_kernel_read_readvariableop-
)savev2_3_dense64_bias_read_readvariableop/
+savev2_4_dense32_kernel_read_readvariableop-
)savev2_4_dense32_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop/
+savev2_5_dense18_kernel_read_readvariableop-
)savev2_5_dense18_bias_read_readvariableop4
0savev2_combined_input_kernel_read_readvariableop2
.savev2_combined_input_bias_read_readvariableop5
1savev2_ensemble_output_kernel_read_readvariableop3
/savev2_ensemble_output_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop5
1savev2_adam_conv3d_5_kernel_m_read_readvariableop3
/savev2_adam_conv3d_5_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_5_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_5_beta_m_read_readvariableop5
1savev2_adam_conv3d_6_kernel_m_read_readvariableop3
/savev2_adam_conv3d_6_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_6_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_6_beta_m_read_readvariableop5
1savev2_adam_conv3d_7_kernel_m_read_readvariableop3
/savev2_adam_conv3d_7_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_7_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_7_beta_m_read_readvariableop5
1savev2_adam_conv3d_8_kernel_m_read_readvariableop3
/savev2_adam_conv3d_8_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_8_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_8_beta_m_read_readvariableop5
1savev2_adam_conv3d_9_kernel_m_read_readvariableop3
/savev2_adam_conv3d_9_bias_m_read_readvariableop6
2savev2_adam_1_dense18_kernel_m_read_readvariableop4
0savev2_adam_1_dense18_bias_m_read_readvariableop6
2savev2_adam_2_dense32_kernel_m_read_readvariableop4
0savev2_adam_2_dense32_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_9_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_9_beta_m_read_readvariableop6
2savev2_adam_3_dense64_kernel_m_read_readvariableop4
0savev2_adam_3_dense64_bias_m_read_readvariableop6
2savev2_adam_4_dense32_kernel_m_read_readvariableop4
0savev2_adam_4_dense32_bias_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableop6
2savev2_adam_5_dense18_kernel_m_read_readvariableop4
0savev2_adam_5_dense18_bias_m_read_readvariableop;
7savev2_adam_combined_input_kernel_m_read_readvariableop9
5savev2_adam_combined_input_bias_m_read_readvariableop<
8savev2_adam_ensemble_output_kernel_m_read_readvariableop:
6savev2_adam_ensemble_output_bias_m_read_readvariableop5
1savev2_adam_conv3d_5_kernel_v_read_readvariableop3
/savev2_adam_conv3d_5_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_5_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_5_beta_v_read_readvariableop5
1savev2_adam_conv3d_6_kernel_v_read_readvariableop3
/savev2_adam_conv3d_6_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_6_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_6_beta_v_read_readvariableop5
1savev2_adam_conv3d_7_kernel_v_read_readvariableop3
/savev2_adam_conv3d_7_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_7_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_7_beta_v_read_readvariableop5
1savev2_adam_conv3d_8_kernel_v_read_readvariableop3
/savev2_adam_conv3d_8_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_8_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_8_beta_v_read_readvariableop5
1savev2_adam_conv3d_9_kernel_v_read_readvariableop3
/savev2_adam_conv3d_9_bias_v_read_readvariableop6
2savev2_adam_1_dense18_kernel_v_read_readvariableop4
0savev2_adam_1_dense18_bias_v_read_readvariableop6
2savev2_adam_2_dense32_kernel_v_read_readvariableop4
0savev2_adam_2_dense32_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_9_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_9_beta_v_read_readvariableop6
2savev2_adam_3_dense64_kernel_v_read_readvariableop4
0savev2_adam_3_dense64_bias_v_read_readvariableop6
2savev2_adam_4_dense32_kernel_v_read_readvariableop4
0savev2_adam_4_dense32_bias_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableop6
2savev2_adam_5_dense18_kernel_v_read_readvariableop4
0savev2_adam_5_dense18_bias_v_read_readvariableop;
7savev2_adam_combined_input_kernel_v_read_readvariableop9
5savev2_adam_combined_input_bias_v_read_readvariableop<
8savev2_adam_ensemble_output_kernel_v_read_readvariableop:
6savev2_adam_ensemble_output_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?G
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?F
value?FB?FB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?5
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_conv3d_5_kernel_read_readvariableop(savev2_conv3d_5_bias_read_readvariableop6savev2_batch_normalization_5_gamma_read_readvariableop5savev2_batch_normalization_5_beta_read_readvariableop<savev2_batch_normalization_5_moving_mean_read_readvariableop@savev2_batch_normalization_5_moving_variance_read_readvariableop*savev2_conv3d_6_kernel_read_readvariableop(savev2_conv3d_6_bias_read_readvariableop6savev2_batch_normalization_6_gamma_read_readvariableop5savev2_batch_normalization_6_beta_read_readvariableop<savev2_batch_normalization_6_moving_mean_read_readvariableop@savev2_batch_normalization_6_moving_variance_read_readvariableop*savev2_conv3d_7_kernel_read_readvariableop(savev2_conv3d_7_bias_read_readvariableop6savev2_batch_normalization_7_gamma_read_readvariableop5savev2_batch_normalization_7_beta_read_readvariableop<savev2_batch_normalization_7_moving_mean_read_readvariableop@savev2_batch_normalization_7_moving_variance_read_readvariableop*savev2_conv3d_8_kernel_read_readvariableop(savev2_conv3d_8_bias_read_readvariableop6savev2_batch_normalization_8_gamma_read_readvariableop5savev2_batch_normalization_8_beta_read_readvariableop<savev2_batch_normalization_8_moving_mean_read_readvariableop@savev2_batch_normalization_8_moving_variance_read_readvariableop*savev2_conv3d_9_kernel_read_readvariableop(savev2_conv3d_9_bias_read_readvariableop+savev2_1_dense18_kernel_read_readvariableop)savev2_1_dense18_bias_read_readvariableop+savev2_2_dense32_kernel_read_readvariableop)savev2_2_dense32_bias_read_readvariableop6savev2_batch_normalization_9_gamma_read_readvariableop5savev2_batch_normalization_9_beta_read_readvariableop<savev2_batch_normalization_9_moving_mean_read_readvariableop@savev2_batch_normalization_9_moving_variance_read_readvariableop+savev2_3_dense64_kernel_read_readvariableop)savev2_3_dense64_bias_read_readvariableop+savev2_4_dense32_kernel_read_readvariableop)savev2_4_dense32_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop+savev2_5_dense18_kernel_read_readvariableop)savev2_5_dense18_bias_read_readvariableop0savev2_combined_input_kernel_read_readvariableop.savev2_combined_input_bias_read_readvariableop1savev2_ensemble_output_kernel_read_readvariableop/savev2_ensemble_output_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop1savev2_adam_conv3d_5_kernel_m_read_readvariableop/savev2_adam_conv3d_5_bias_m_read_readvariableop=savev2_adam_batch_normalization_5_gamma_m_read_readvariableop<savev2_adam_batch_normalization_5_beta_m_read_readvariableop1savev2_adam_conv3d_6_kernel_m_read_readvariableop/savev2_adam_conv3d_6_bias_m_read_readvariableop=savev2_adam_batch_normalization_6_gamma_m_read_readvariableop<savev2_adam_batch_normalization_6_beta_m_read_readvariableop1savev2_adam_conv3d_7_kernel_m_read_readvariableop/savev2_adam_conv3d_7_bias_m_read_readvariableop=savev2_adam_batch_normalization_7_gamma_m_read_readvariableop<savev2_adam_batch_normalization_7_beta_m_read_readvariableop1savev2_adam_conv3d_8_kernel_m_read_readvariableop/savev2_adam_conv3d_8_bias_m_read_readvariableop=savev2_adam_batch_normalization_8_gamma_m_read_readvariableop<savev2_adam_batch_normalization_8_beta_m_read_readvariableop1savev2_adam_conv3d_9_kernel_m_read_readvariableop/savev2_adam_conv3d_9_bias_m_read_readvariableop2savev2_adam_1_dense18_kernel_m_read_readvariableop0savev2_adam_1_dense18_bias_m_read_readvariableop2savev2_adam_2_dense32_kernel_m_read_readvariableop0savev2_adam_2_dense32_bias_m_read_readvariableop=savev2_adam_batch_normalization_9_gamma_m_read_readvariableop<savev2_adam_batch_normalization_9_beta_m_read_readvariableop2savev2_adam_3_dense64_kernel_m_read_readvariableop0savev2_adam_3_dense64_bias_m_read_readvariableop2savev2_adam_4_dense32_kernel_m_read_readvariableop0savev2_adam_4_dense32_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop2savev2_adam_5_dense18_kernel_m_read_readvariableop0savev2_adam_5_dense18_bias_m_read_readvariableop7savev2_adam_combined_input_kernel_m_read_readvariableop5savev2_adam_combined_input_bias_m_read_readvariableop8savev2_adam_ensemble_output_kernel_m_read_readvariableop6savev2_adam_ensemble_output_bias_m_read_readvariableop1savev2_adam_conv3d_5_kernel_v_read_readvariableop/savev2_adam_conv3d_5_bias_v_read_readvariableop=savev2_adam_batch_normalization_5_gamma_v_read_readvariableop<savev2_adam_batch_normalization_5_beta_v_read_readvariableop1savev2_adam_conv3d_6_kernel_v_read_readvariableop/savev2_adam_conv3d_6_bias_v_read_readvariableop=savev2_adam_batch_normalization_6_gamma_v_read_readvariableop<savev2_adam_batch_normalization_6_beta_v_read_readvariableop1savev2_adam_conv3d_7_kernel_v_read_readvariableop/savev2_adam_conv3d_7_bias_v_read_readvariableop=savev2_adam_batch_normalization_7_gamma_v_read_readvariableop<savev2_adam_batch_normalization_7_beta_v_read_readvariableop1savev2_adam_conv3d_8_kernel_v_read_readvariableop/savev2_adam_conv3d_8_bias_v_read_readvariableop=savev2_adam_batch_normalization_8_gamma_v_read_readvariableop<savev2_adam_batch_normalization_8_beta_v_read_readvariableop1savev2_adam_conv3d_9_kernel_v_read_readvariableop/savev2_adam_conv3d_9_bias_v_read_readvariableop2savev2_adam_1_dense18_kernel_v_read_readvariableop0savev2_adam_1_dense18_bias_v_read_readvariableop2savev2_adam_2_dense32_kernel_v_read_readvariableop0savev2_adam_2_dense32_bias_v_read_readvariableop=savev2_adam_batch_normalization_9_gamma_v_read_readvariableop<savev2_adam_batch_normalization_9_beta_v_read_readvariableop2savev2_adam_3_dense64_kernel_v_read_readvariableop0savev2_adam_3_dense64_bias_v_read_readvariableop2savev2_adam_4_dense32_kernel_v_read_readvariableop0savev2_adam_4_dense32_bias_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableop2savev2_adam_5_dense18_kernel_v_read_readvariableop0savev2_adam_5_dense18_bias_v_read_readvariableop7savev2_adam_combined_input_kernel_v_read_readvariableop5savev2_adam_combined_input_bias_v_read_readvariableop8savev2_adam_ensemble_output_kernel_v_read_readvariableop6savev2_adam_ensemble_output_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *?
dtypes?
?2	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*?
_input_shapes?
?: : : : : : : :  : : : : : : @:@:@:@:@:@:@?:?:?:?:?:?:??:?:
:: : :?:?:?:?: @:@:@ : :
??:?: ::	?	:	:	:: : : : : : : : : : : : :  : : : : @:@:@:@:@?:?:?:?:??:?:
:: : :?:?: @:@:@ : :
??:?: ::	?	:	:	:: : : : :  : : : : @:@:@:@:@?:?:?:?:??:?:
:: : :?:?: @:@:@ : :
??:?: ::	?	:	:	:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:0,
*
_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :0,
*
_output_shapes
:  : 

_output_shapes
: : 	

_output_shapes
: : 


_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :0,
*
_output_shapes
: @: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:1-
+
_output_shapes
:@?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:2.
,
_output_shapes
:??:!

_output_shapes	
:?:$ 

_output_shapes

:
: 

_output_shapes
::$ 

_output_shapes

: : 

_output_shapes
: :!

_output_shapes	
:?:! 

_output_shapes	
:?:!!

_output_shapes	
:?:!"

_output_shapes	
:?:$# 

_output_shapes

: @: $

_output_shapes
:@:$% 

_output_shapes

:@ : &

_output_shapes
: :&'"
 
_output_shapes
:
??:!(

_output_shapes	
:?:$) 

_output_shapes

: : *

_output_shapes
::%+!

_output_shapes
:	?	: ,

_output_shapes
:	:$- 

_output_shapes

:	: .

_output_shapes
::/

_output_shapes
: :0

_output_shapes
: :1

_output_shapes
: :2

_output_shapes
: :3

_output_shapes
: :4

_output_shapes
: :5

_output_shapes
: :6

_output_shapes
: :07,
*
_output_shapes
: : 8

_output_shapes
: : 9

_output_shapes
: : :

_output_shapes
: :0;,
*
_output_shapes
:  : <

_output_shapes
: : =

_output_shapes
: : >

_output_shapes
: :0?,
*
_output_shapes
: @: @

_output_shapes
:@: A

_output_shapes
:@: B

_output_shapes
:@:1C-
+
_output_shapes
:@?:!D

_output_shapes	
:?:!E

_output_shapes	
:?:!F

_output_shapes	
:?:2G.
,
_output_shapes
:??:!H

_output_shapes	
:?:$I 

_output_shapes

:
: J

_output_shapes
::$K 

_output_shapes

: : L

_output_shapes
: :!M

_output_shapes	
:?:!N

_output_shapes	
:?:$O 

_output_shapes

: @: P

_output_shapes
:@:$Q 

_output_shapes

:@ : R

_output_shapes
: :&S"
 
_output_shapes
:
??:!T

_output_shapes	
:?:$U 

_output_shapes

: : V

_output_shapes
::%W!

_output_shapes
:	?	: X

_output_shapes
:	:$Y 

_output_shapes

:	: Z

_output_shapes
::0[,
*
_output_shapes
: : \

_output_shapes
: : ]

_output_shapes
: : ^

_output_shapes
: :0_,
*
_output_shapes
:  : `

_output_shapes
: : a

_output_shapes
: : b

_output_shapes
: :0c,
*
_output_shapes
: @: d

_output_shapes
:@: e

_output_shapes
:@: f

_output_shapes
:@:1g-
+
_output_shapes
:@?:!h

_output_shapes	
:?:!i

_output_shapes	
:?:!j

_output_shapes	
:?:2k.
,
_output_shapes
:??:!l

_output_shapes	
:?:$m 

_output_shapes

:
: n

_output_shapes
::$o 

_output_shapes

: : p

_output_shapes
: :!q

_output_shapes	
:?:!r

_output_shapes	
:?:$s 

_output_shapes

: @: t

_output_shapes
:@:$u 

_output_shapes

:@ : v

_output_shapes
: :&w"
 
_output_shapes
:
??:!x

_output_shapes	
:?:$y 

_output_shapes

: : z

_output_shapes
::%{!

_output_shapes
:	?	: |

_output_shapes
:	:$} 

_output_shapes

:	: ~

_output_shapes
::

_output_shapes
: 
?
?
*__inference_conv3d_7_layer_call_fn_9257918

inputs%
unknown: @
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv3d_7_layer_call_and_return_conditional_losses_92554552
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:?????????@@@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@@ : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:?????????@@ 
 
_user_specified_nameinputs
?
?
1__inference_ensemble_output_layer_call_fn_9258612

inputs
unknown:	
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_ensemble_output_layer_call_and_return_conditional_losses_92557472
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????	: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_9254807

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*j
_output_shapesX
V:8???????????????????????????????????? : : : : :*
data_formatNDHWC*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*N
_output_shapes<
::8???????????????????????????????????? 2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8???????????????????????????????????? : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:v r
N
_output_shapes<
::8???????????????????????????????????? 
 
_user_specified_nameinputs
?
?
F__inference_5_dense18_layer_call_and_return_conditional_losses_9258543

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd^
	LeakyRelu	LeakyReluBiasAdd:output:0*'
_output_shapes
:?????????2
	LeakyRelur
IdentityIdentityLeakyRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
M
1__inference_max_pooling3d_8_layer_call_fn_9258103

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????@@?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling3d_8_layer_call_and_return_conditional_losses_92555152
PartitionedCally
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :?????????@@?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????@@?:\ X
4
_output_shapes"
 :?????????@@?
 
_user_specified_nameinputs
?
?
E__inference_conv3d_6_layer_call_and_return_conditional_losses_9255405

inputs<
conv3d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:  *
dtype02
Conv3D/ReadVariableOp?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@@ *
paddingSAME*
strides	
2
Conv3D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@@ 2	
BiasAddj
	LeakyRelu	LeakyReluBiasAdd:output:0*3
_output_shapes!
:?????????@@ 2
	LeakyRelu~
IdentityIdentityLeakyRelu:activations:0^NoOp*
T0*3
_output_shapes!
:?????????@@ 2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@@ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:?????????@@ 
 
_user_specified_nameinputs
?
M
1__inference_max_pooling3d_6_layer_call_fn_9257775

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@@ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling3d_6_layer_call_and_return_conditional_losses_92554152
PartitionedCallx
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:?????????@@ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@ :[ W
3
_output_shapes!
:?????????@@ 
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_9258387

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*o
_output_shapes]
[:9?????????????????????????????????????:?:?:?:?:*
data_formatNDHWC*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*O
_output_shapes=
;:9?????????????????????????????????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:9?????????????????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:w s
O
_output_shapes=
;:9?????????????????????????????????????
 
_user_specified_nameinputs
?
h
L__inference_max_pooling3d_5_layer_call_and_return_conditional_losses_9254580

inputs
identity?
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????*
ksize	
*
paddingVALID*
strides	
2
	MaxPool3D?
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A?????????????????????????????????????????????: {
W
_output_shapesE
C:A?????????????????????????????????????????????
 
_user_specified_nameinputs
?
?
+__inference_2_dense32_layer_call_fn_9258306

inputs
unknown: 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_2_dense32_layer_call_and_return_conditional_losses_92556222
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
M
1__inference_max_pooling3d_5_layer_call_fn_9257611

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@@ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling3d_5_layer_call_and_return_conditional_losses_92553652
PartitionedCallx
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:?????????@@ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:??????????? :] Y
5
_output_shapes#
!:??????????? 
 
_user_specified_nameinputs
?
e
F__inference_dropout_1_layer_call_and_return_conditional_losses_9255896

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniformg
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *???>2
dropout/Const_1?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/Const_1:output:0*
T0*(
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
)__inference_dense_1_layer_call_fn_9258512

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_92556632
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
ۦ
?.
"__inference__wrapped_model_9254571

svm_inputs
img3d_inputsP
2ensemble4d_conv3d_5_conv3d_readvariableop_resource: A
3ensemble4d_conv3d_5_biasadd_readvariableop_resource: F
8ensemble4d_batch_normalization_5_readvariableop_resource: H
:ensemble4d_batch_normalization_5_readvariableop_1_resource: W
Iensemble4d_batch_normalization_5_fusedbatchnormv3_readvariableop_resource: Y
Kensemble4d_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource: P
2ensemble4d_conv3d_6_conv3d_readvariableop_resource:  A
3ensemble4d_conv3d_6_biasadd_readvariableop_resource: F
8ensemble4d_batch_normalization_6_readvariableop_resource: H
:ensemble4d_batch_normalization_6_readvariableop_1_resource: W
Iensemble4d_batch_normalization_6_fusedbatchnormv3_readvariableop_resource: Y
Kensemble4d_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource: P
2ensemble4d_conv3d_7_conv3d_readvariableop_resource: @A
3ensemble4d_conv3d_7_biasadd_readvariableop_resource:@F
8ensemble4d_batch_normalization_7_readvariableop_resource:@H
:ensemble4d_batch_normalization_7_readvariableop_1_resource:@W
Iensemble4d_batch_normalization_7_fusedbatchnormv3_readvariableop_resource:@Y
Kensemble4d_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource:@Q
2ensemble4d_conv3d_8_conv3d_readvariableop_resource:@?B
3ensemble4d_conv3d_8_biasadd_readvariableop_resource:	?G
8ensemble4d_batch_normalization_8_readvariableop_resource:	?I
:ensemble4d_batch_normalization_8_readvariableop_1_resource:	?X
Iensemble4d_batch_normalization_8_fusedbatchnormv3_readvariableop_resource:	?Z
Kensemble4d_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource:	?R
2ensemble4d_conv3d_9_conv3d_readvariableop_resource:??B
3ensemble4d_conv3d_9_biasadd_readvariableop_resource:	?E
3ensemble4d_1_dense18_matmul_readvariableop_resource:
B
4ensemble4d_1_dense18_biasadd_readvariableop_resource:G
8ensemble4d_batch_normalization_9_readvariableop_resource:	?I
:ensemble4d_batch_normalization_9_readvariableop_1_resource:	?X
Iensemble4d_batch_normalization_9_fusedbatchnormv3_readvariableop_resource:	?Z
Kensemble4d_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource:	?E
3ensemble4d_2_dense32_matmul_readvariableop_resource: B
4ensemble4d_2_dense32_biasadd_readvariableop_resource: E
3ensemble4d_3_dense64_matmul_readvariableop_resource: @B
4ensemble4d_3_dense64_biasadd_readvariableop_resource:@E
1ensemble4d_dense_1_matmul_readvariableop_resource:
??A
2ensemble4d_dense_1_biasadd_readvariableop_resource:	?E
3ensemble4d_4_dense32_matmul_readvariableop_resource:@ B
4ensemble4d_4_dense32_biasadd_readvariableop_resource: E
3ensemble4d_5_dense18_matmul_readvariableop_resource: B
4ensemble4d_5_dense18_biasadd_readvariableop_resource:K
8ensemble4d_combined_input_matmul_readvariableop_resource:	?	G
9ensemble4d_combined_input_biasadd_readvariableop_resource:	K
9ensemble4d_ensemble_output_matmul_readvariableop_resource:	H
:ensemble4d_ensemble_output_biasadd_readvariableop_resource:
identity??+ensemble4d/1_dense18/BiasAdd/ReadVariableOp?*ensemble4d/1_dense18/MatMul/ReadVariableOp?+ensemble4d/2_dense32/BiasAdd/ReadVariableOp?*ensemble4d/2_dense32/MatMul/ReadVariableOp?+ensemble4d/3_dense64/BiasAdd/ReadVariableOp?*ensemble4d/3_dense64/MatMul/ReadVariableOp?+ensemble4d/4_dense32/BiasAdd/ReadVariableOp?*ensemble4d/4_dense32/MatMul/ReadVariableOp?+ensemble4d/5_dense18/BiasAdd/ReadVariableOp?*ensemble4d/5_dense18/MatMul/ReadVariableOp?@ensemble4d/batch_normalization_5/FusedBatchNormV3/ReadVariableOp?Bensemble4d/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?/ensemble4d/batch_normalization_5/ReadVariableOp?1ensemble4d/batch_normalization_5/ReadVariableOp_1?@ensemble4d/batch_normalization_6/FusedBatchNormV3/ReadVariableOp?Bensemble4d/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?/ensemble4d/batch_normalization_6/ReadVariableOp?1ensemble4d/batch_normalization_6/ReadVariableOp_1?@ensemble4d/batch_normalization_7/FusedBatchNormV3/ReadVariableOp?Bensemble4d/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1?/ensemble4d/batch_normalization_7/ReadVariableOp?1ensemble4d/batch_normalization_7/ReadVariableOp_1?@ensemble4d/batch_normalization_8/FusedBatchNormV3/ReadVariableOp?Bensemble4d/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?/ensemble4d/batch_normalization_8/ReadVariableOp?1ensemble4d/batch_normalization_8/ReadVariableOp_1?@ensemble4d/batch_normalization_9/FusedBatchNormV3/ReadVariableOp?Bensemble4d/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1?/ensemble4d/batch_normalization_9/ReadVariableOp?1ensemble4d/batch_normalization_9/ReadVariableOp_1?0ensemble4d/combined_input/BiasAdd/ReadVariableOp?/ensemble4d/combined_input/MatMul/ReadVariableOp?*ensemble4d/conv3d_5/BiasAdd/ReadVariableOp?)ensemble4d/conv3d_5/Conv3D/ReadVariableOp?*ensemble4d/conv3d_6/BiasAdd/ReadVariableOp?)ensemble4d/conv3d_6/Conv3D/ReadVariableOp?*ensemble4d/conv3d_7/BiasAdd/ReadVariableOp?)ensemble4d/conv3d_7/Conv3D/ReadVariableOp?*ensemble4d/conv3d_8/BiasAdd/ReadVariableOp?)ensemble4d/conv3d_8/Conv3D/ReadVariableOp?*ensemble4d/conv3d_9/BiasAdd/ReadVariableOp?)ensemble4d/conv3d_9/Conv3D/ReadVariableOp?)ensemble4d/dense_1/BiasAdd/ReadVariableOp?(ensemble4d/dense_1/MatMul/ReadVariableOp?1ensemble4d/ensemble_output/BiasAdd/ReadVariableOp?0ensemble4d/ensemble_output/MatMul/ReadVariableOp?
)ensemble4d/conv3d_5/Conv3D/ReadVariableOpReadVariableOp2ensemble4d_conv3d_5_conv3d_readvariableop_resource**
_output_shapes
: *
dtype02+
)ensemble4d/conv3d_5/Conv3D/ReadVariableOp?
ensemble4d/conv3d_5/Conv3DConv3Dimg3d_inputs1ensemble4d/conv3d_5/Conv3D/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:??????????? *
paddingSAME*
strides	
2
ensemble4d/conv3d_5/Conv3D?
*ensemble4d/conv3d_5/BiasAdd/ReadVariableOpReadVariableOp3ensemble4d_conv3d_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02,
*ensemble4d/conv3d_5/BiasAdd/ReadVariableOp?
ensemble4d/conv3d_5/BiasAddBiasAdd#ensemble4d/conv3d_5/Conv3D:output:02ensemble4d/conv3d_5/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:??????????? 2
ensemble4d/conv3d_5/BiasAdd?
ensemble4d/conv3d_5/LeakyRelu	LeakyRelu$ensemble4d/conv3d_5/BiasAdd:output:0*5
_output_shapes#
!:??????????? 2
ensemble4d/conv3d_5/LeakyRelu?
$ensemble4d/max_pooling3d_5/MaxPool3D	MaxPool3D+ensemble4d/conv3d_5/LeakyRelu:activations:0*
T0*3
_output_shapes!
:?????????@@ *
ksize	
*
paddingVALID*
strides	
2&
$ensemble4d/max_pooling3d_5/MaxPool3D?
/ensemble4d/batch_normalization_5/ReadVariableOpReadVariableOp8ensemble4d_batch_normalization_5_readvariableop_resource*
_output_shapes
: *
dtype021
/ensemble4d/batch_normalization_5/ReadVariableOp?
1ensemble4d/batch_normalization_5/ReadVariableOp_1ReadVariableOp:ensemble4d_batch_normalization_5_readvariableop_1_resource*
_output_shapes
: *
dtype023
1ensemble4d/batch_normalization_5/ReadVariableOp_1?
@ensemble4d/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpIensemble4d_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02B
@ensemble4d/batch_normalization_5/FusedBatchNormV3/ReadVariableOp?
Bensemble4d/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpKensemble4d_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02D
Bensemble4d/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?
1ensemble4d/batch_normalization_5/FusedBatchNormV3FusedBatchNormV3-ensemble4d/max_pooling3d_5/MaxPool3D:output:07ensemble4d/batch_normalization_5/ReadVariableOp:value:09ensemble4d/batch_normalization_5/ReadVariableOp_1:value:0Hensemble4d/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0Jensemble4d/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*O
_output_shapes=
;:?????????@@ : : : : :*
data_formatNDHWC*
epsilon%o?:*
is_training( 23
1ensemble4d/batch_normalization_5/FusedBatchNormV3?
)ensemble4d/conv3d_6/Conv3D/ReadVariableOpReadVariableOp2ensemble4d_conv3d_6_conv3d_readvariableop_resource**
_output_shapes
:  *
dtype02+
)ensemble4d/conv3d_6/Conv3D/ReadVariableOp?
ensemble4d/conv3d_6/Conv3DConv3D5ensemble4d/batch_normalization_5/FusedBatchNormV3:y:01ensemble4d/conv3d_6/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@@ *
paddingSAME*
strides	
2
ensemble4d/conv3d_6/Conv3D?
*ensemble4d/conv3d_6/BiasAdd/ReadVariableOpReadVariableOp3ensemble4d_conv3d_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02,
*ensemble4d/conv3d_6/BiasAdd/ReadVariableOp?
ensemble4d/conv3d_6/BiasAddBiasAdd#ensemble4d/conv3d_6/Conv3D:output:02ensemble4d/conv3d_6/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@@ 2
ensemble4d/conv3d_6/BiasAdd?
ensemble4d/conv3d_6/LeakyRelu	LeakyRelu$ensemble4d/conv3d_6/BiasAdd:output:0*3
_output_shapes!
:?????????@@ 2
ensemble4d/conv3d_6/LeakyRelu?
$ensemble4d/max_pooling3d_6/MaxPool3D	MaxPool3D+ensemble4d/conv3d_6/LeakyRelu:activations:0*
T0*3
_output_shapes!
:?????????@@ *
ksize	
*
paddingVALID*
strides	
2&
$ensemble4d/max_pooling3d_6/MaxPool3D?
/ensemble4d/batch_normalization_6/ReadVariableOpReadVariableOp8ensemble4d_batch_normalization_6_readvariableop_resource*
_output_shapes
: *
dtype021
/ensemble4d/batch_normalization_6/ReadVariableOp?
1ensemble4d/batch_normalization_6/ReadVariableOp_1ReadVariableOp:ensemble4d_batch_normalization_6_readvariableop_1_resource*
_output_shapes
: *
dtype023
1ensemble4d/batch_normalization_6/ReadVariableOp_1?
@ensemble4d/batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOpIensemble4d_batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02B
@ensemble4d/batch_normalization_6/FusedBatchNormV3/ReadVariableOp?
Bensemble4d/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpKensemble4d_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02D
Bensemble4d/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?
1ensemble4d/batch_normalization_6/FusedBatchNormV3FusedBatchNormV3-ensemble4d/max_pooling3d_6/MaxPool3D:output:07ensemble4d/batch_normalization_6/ReadVariableOp:value:09ensemble4d/batch_normalization_6/ReadVariableOp_1:value:0Hensemble4d/batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0Jensemble4d/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*O
_output_shapes=
;:?????????@@ : : : : :*
data_formatNDHWC*
epsilon%o?:*
is_training( 23
1ensemble4d/batch_normalization_6/FusedBatchNormV3?
)ensemble4d/conv3d_7/Conv3D/ReadVariableOpReadVariableOp2ensemble4d_conv3d_7_conv3d_readvariableop_resource**
_output_shapes
: @*
dtype02+
)ensemble4d/conv3d_7/Conv3D/ReadVariableOp?
ensemble4d/conv3d_7/Conv3DConv3D5ensemble4d/batch_normalization_6/FusedBatchNormV3:y:01ensemble4d/conv3d_7/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@@@*
paddingSAME*
strides	
2
ensemble4d/conv3d_7/Conv3D?
*ensemble4d/conv3d_7/BiasAdd/ReadVariableOpReadVariableOp3ensemble4d_conv3d_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*ensemble4d/conv3d_7/BiasAdd/ReadVariableOp?
ensemble4d/conv3d_7/BiasAddBiasAdd#ensemble4d/conv3d_7/Conv3D:output:02ensemble4d/conv3d_7/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@@@2
ensemble4d/conv3d_7/BiasAdd?
ensemble4d/conv3d_7/LeakyRelu	LeakyRelu$ensemble4d/conv3d_7/BiasAdd:output:0*3
_output_shapes!
:?????????@@@2
ensemble4d/conv3d_7/LeakyRelu?
$ensemble4d/max_pooling3d_7/MaxPool3D	MaxPool3D+ensemble4d/conv3d_7/LeakyRelu:activations:0*
T0*3
_output_shapes!
:?????????@@@*
ksize	
*
paddingVALID*
strides	
2&
$ensemble4d/max_pooling3d_7/MaxPool3D?
/ensemble4d/batch_normalization_7/ReadVariableOpReadVariableOp8ensemble4d_batch_normalization_7_readvariableop_resource*
_output_shapes
:@*
dtype021
/ensemble4d/batch_normalization_7/ReadVariableOp?
1ensemble4d/batch_normalization_7/ReadVariableOp_1ReadVariableOp:ensemble4d_batch_normalization_7_readvariableop_1_resource*
_output_shapes
:@*
dtype023
1ensemble4d/batch_normalization_7/ReadVariableOp_1?
@ensemble4d/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpIensemble4d_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02B
@ensemble4d/batch_normalization_7/FusedBatchNormV3/ReadVariableOp?
Bensemble4d/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpKensemble4d_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02D
Bensemble4d/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1?
1ensemble4d/batch_normalization_7/FusedBatchNormV3FusedBatchNormV3-ensemble4d/max_pooling3d_7/MaxPool3D:output:07ensemble4d/batch_normalization_7/ReadVariableOp:value:09ensemble4d/batch_normalization_7/ReadVariableOp_1:value:0Hensemble4d/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0Jensemble4d/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*O
_output_shapes=
;:?????????@@@:@:@:@:@:*
data_formatNDHWC*
epsilon%o?:*
is_training( 23
1ensemble4d/batch_normalization_7/FusedBatchNormV3?
)ensemble4d/conv3d_8/Conv3D/ReadVariableOpReadVariableOp2ensemble4d_conv3d_8_conv3d_readvariableop_resource*+
_output_shapes
:@?*
dtype02+
)ensemble4d/conv3d_8/Conv3D/ReadVariableOp?
ensemble4d/conv3d_8/Conv3DConv3D5ensemble4d/batch_normalization_7/FusedBatchNormV3:y:01ensemble4d/conv3d_8/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :?????????@@?*
paddingSAME*
strides	
2
ensemble4d/conv3d_8/Conv3D?
*ensemble4d/conv3d_8/BiasAdd/ReadVariableOpReadVariableOp3ensemble4d_conv3d_8_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02,
*ensemble4d/conv3d_8/BiasAdd/ReadVariableOp?
ensemble4d/conv3d_8/BiasAddBiasAdd#ensemble4d/conv3d_8/Conv3D:output:02ensemble4d/conv3d_8/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :?????????@@?2
ensemble4d/conv3d_8/BiasAdd?
ensemble4d/conv3d_8/LeakyRelu	LeakyRelu$ensemble4d/conv3d_8/BiasAdd:output:0*4
_output_shapes"
 :?????????@@?2
ensemble4d/conv3d_8/LeakyRelu?
$ensemble4d/max_pooling3d_8/MaxPool3D	MaxPool3D+ensemble4d/conv3d_8/LeakyRelu:activations:0*
T0*4
_output_shapes"
 :?????????@@?*
ksize	
*
paddingVALID*
strides	
2&
$ensemble4d/max_pooling3d_8/MaxPool3D?
/ensemble4d/batch_normalization_8/ReadVariableOpReadVariableOp8ensemble4d_batch_normalization_8_readvariableop_resource*
_output_shapes	
:?*
dtype021
/ensemble4d/batch_normalization_8/ReadVariableOp?
1ensemble4d/batch_normalization_8/ReadVariableOp_1ReadVariableOp:ensemble4d_batch_normalization_8_readvariableop_1_resource*
_output_shapes	
:?*
dtype023
1ensemble4d/batch_normalization_8/ReadVariableOp_1?
@ensemble4d/batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOpIensemble4d_batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02B
@ensemble4d/batch_normalization_8/FusedBatchNormV3/ReadVariableOp?
Bensemble4d/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpKensemble4d_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02D
Bensemble4d/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?
1ensemble4d/batch_normalization_8/FusedBatchNormV3FusedBatchNormV3-ensemble4d/max_pooling3d_8/MaxPool3D:output:07ensemble4d/batch_normalization_8/ReadVariableOp:value:09ensemble4d/batch_normalization_8/ReadVariableOp_1:value:0Hensemble4d/batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0Jensemble4d/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*T
_output_shapesB
@:?????????@@?:?:?:?:?:*
data_formatNDHWC*
epsilon%o?:*
is_training( 23
1ensemble4d/batch_normalization_8/FusedBatchNormV3?
)ensemble4d/conv3d_9/Conv3D/ReadVariableOpReadVariableOp2ensemble4d_conv3d_9_conv3d_readvariableop_resource*,
_output_shapes
:??*
dtype02+
)ensemble4d/conv3d_9/Conv3D/ReadVariableOp?
ensemble4d/conv3d_9/Conv3DConv3D5ensemble4d/batch_normalization_8/FusedBatchNormV3:y:01ensemble4d/conv3d_9/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :?????????@@?*
paddingSAME*
strides	
2
ensemble4d/conv3d_9/Conv3D?
*ensemble4d/conv3d_9/BiasAdd/ReadVariableOpReadVariableOp3ensemble4d_conv3d_9_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02,
*ensemble4d/conv3d_9/BiasAdd/ReadVariableOp?
ensemble4d/conv3d_9/BiasAddBiasAdd#ensemble4d/conv3d_9/Conv3D:output:02ensemble4d/conv3d_9/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :?????????@@?2
ensemble4d/conv3d_9/BiasAdd?
ensemble4d/conv3d_9/LeakyRelu	LeakyRelu$ensemble4d/conv3d_9/BiasAdd:output:0*4
_output_shapes"
 :?????????@@?2
ensemble4d/conv3d_9/LeakyRelu?
$ensemble4d/max_pooling3d_9/MaxPool3D	MaxPool3D+ensemble4d/conv3d_9/LeakyRelu:activations:0*
T0*4
_output_shapes"
 :?????????@@?*
ksize	
*
paddingVALID*
strides	
2&
$ensemble4d/max_pooling3d_9/MaxPool3D?
*ensemble4d/1_dense18/MatMul/ReadVariableOpReadVariableOp3ensemble4d_1_dense18_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02,
*ensemble4d/1_dense18/MatMul/ReadVariableOp?
ensemble4d/1_dense18/MatMulMatMul
svm_inputs2ensemble4d/1_dense18/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
ensemble4d/1_dense18/MatMul?
+ensemble4d/1_dense18/BiasAdd/ReadVariableOpReadVariableOp4ensemble4d_1_dense18_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+ensemble4d/1_dense18/BiasAdd/ReadVariableOp?
ensemble4d/1_dense18/BiasAddBiasAdd%ensemble4d/1_dense18/MatMul:product:03ensemble4d/1_dense18/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
ensemble4d/1_dense18/BiasAdd?
ensemble4d/1_dense18/LeakyRelu	LeakyRelu%ensemble4d/1_dense18/BiasAdd:output:0*'
_output_shapes
:?????????2 
ensemble4d/1_dense18/LeakyRelu?
/ensemble4d/batch_normalization_9/ReadVariableOpReadVariableOp8ensemble4d_batch_normalization_9_readvariableop_resource*
_output_shapes	
:?*
dtype021
/ensemble4d/batch_normalization_9/ReadVariableOp?
1ensemble4d/batch_normalization_9/ReadVariableOp_1ReadVariableOp:ensemble4d_batch_normalization_9_readvariableop_1_resource*
_output_shapes	
:?*
dtype023
1ensemble4d/batch_normalization_9/ReadVariableOp_1?
@ensemble4d/batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOpIensemble4d_batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02B
@ensemble4d/batch_normalization_9/FusedBatchNormV3/ReadVariableOp?
Bensemble4d/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpKensemble4d_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02D
Bensemble4d/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1?
1ensemble4d/batch_normalization_9/FusedBatchNormV3FusedBatchNormV3-ensemble4d/max_pooling3d_9/MaxPool3D:output:07ensemble4d/batch_normalization_9/ReadVariableOp:value:09ensemble4d/batch_normalization_9/ReadVariableOp_1:value:0Hensemble4d/batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0Jensemble4d/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*T
_output_shapesB
@:?????????@@?:?:?:?:?:*
data_formatNDHWC*
epsilon%o?:*
is_training( 23
1ensemble4d/batch_normalization_9/FusedBatchNormV3?
*ensemble4d/2_dense32/MatMul/ReadVariableOpReadVariableOp3ensemble4d_2_dense32_matmul_readvariableop_resource*
_output_shapes

: *
dtype02,
*ensemble4d/2_dense32/MatMul/ReadVariableOp?
ensemble4d/2_dense32/MatMulMatMul,ensemble4d/1_dense18/LeakyRelu:activations:02ensemble4d/2_dense32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
ensemble4d/2_dense32/MatMul?
+ensemble4d/2_dense32/BiasAdd/ReadVariableOpReadVariableOp4ensemble4d_2_dense32_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+ensemble4d/2_dense32/BiasAdd/ReadVariableOp?
ensemble4d/2_dense32/BiasAddBiasAdd%ensemble4d/2_dense32/MatMul:product:03ensemble4d/2_dense32/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
ensemble4d/2_dense32/BiasAdd?
ensemble4d/2_dense32/LeakyRelu	LeakyRelu%ensemble4d/2_dense32/BiasAdd:output:0*'
_output_shapes
:????????? 2 
ensemble4d/2_dense32/LeakyRelu?
<ensemble4d/global_average_pooling3d_1/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2>
<ensemble4d/global_average_pooling3d_1/Mean/reduction_indices?
*ensemble4d/global_average_pooling3d_1/MeanMean5ensemble4d/batch_normalization_9/FusedBatchNormV3:y:0Eensemble4d/global_average_pooling3d_1/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2,
*ensemble4d/global_average_pooling3d_1/Mean?
*ensemble4d/3_dense64/MatMul/ReadVariableOpReadVariableOp3ensemble4d_3_dense64_matmul_readvariableop_resource*
_output_shapes

: @*
dtype02,
*ensemble4d/3_dense64/MatMul/ReadVariableOp?
ensemble4d/3_dense64/MatMulMatMul,ensemble4d/2_dense32/LeakyRelu:activations:02ensemble4d/3_dense64/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
ensemble4d/3_dense64/MatMul?
+ensemble4d/3_dense64/BiasAdd/ReadVariableOpReadVariableOp4ensemble4d_3_dense64_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+ensemble4d/3_dense64/BiasAdd/ReadVariableOp?
ensemble4d/3_dense64/BiasAddBiasAdd%ensemble4d/3_dense64/MatMul:product:03ensemble4d/3_dense64/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
ensemble4d/3_dense64/BiasAdd?
ensemble4d/3_dense64/LeakyRelu	LeakyRelu%ensemble4d/3_dense64/BiasAdd:output:0*'
_output_shapes
:?????????@2 
ensemble4d/3_dense64/LeakyRelu?
(ensemble4d/dense_1/MatMul/ReadVariableOpReadVariableOp1ensemble4d_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02*
(ensemble4d/dense_1/MatMul/ReadVariableOp?
ensemble4d/dense_1/MatMulMatMul3ensemble4d/global_average_pooling3d_1/Mean:output:00ensemble4d/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
ensemble4d/dense_1/MatMul?
)ensemble4d/dense_1/BiasAdd/ReadVariableOpReadVariableOp2ensemble4d_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)ensemble4d/dense_1/BiasAdd/ReadVariableOp?
ensemble4d/dense_1/BiasAddBiasAdd#ensemble4d/dense_1/MatMul:product:01ensemble4d/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
ensemble4d/dense_1/BiasAdd?
ensemble4d/dense_1/LeakyRelu	LeakyRelu#ensemble4d/dense_1/BiasAdd:output:0*(
_output_shapes
:??????????2
ensemble4d/dense_1/LeakyRelu?
*ensemble4d/4_dense32/MatMul/ReadVariableOpReadVariableOp3ensemble4d_4_dense32_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02,
*ensemble4d/4_dense32/MatMul/ReadVariableOp?
ensemble4d/4_dense32/MatMulMatMul,ensemble4d/3_dense64/LeakyRelu:activations:02ensemble4d/4_dense32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
ensemble4d/4_dense32/MatMul?
+ensemble4d/4_dense32/BiasAdd/ReadVariableOpReadVariableOp4ensemble4d_4_dense32_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+ensemble4d/4_dense32/BiasAdd/ReadVariableOp?
ensemble4d/4_dense32/BiasAddBiasAdd%ensemble4d/4_dense32/MatMul:product:03ensemble4d/4_dense32/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
ensemble4d/4_dense32/BiasAdd?
ensemble4d/4_dense32/LeakyRelu	LeakyRelu%ensemble4d/4_dense32/BiasAdd:output:0*'
_output_shapes
:????????? 2 
ensemble4d/4_dense32/LeakyRelu?
*ensemble4d/5_dense18/MatMul/ReadVariableOpReadVariableOp3ensemble4d_5_dense18_matmul_readvariableop_resource*
_output_shapes

: *
dtype02,
*ensemble4d/5_dense18/MatMul/ReadVariableOp?
ensemble4d/5_dense18/MatMulMatMul,ensemble4d/4_dense32/LeakyRelu:activations:02ensemble4d/5_dense18/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
ensemble4d/5_dense18/MatMul?
+ensemble4d/5_dense18/BiasAdd/ReadVariableOpReadVariableOp4ensemble4d_5_dense18_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+ensemble4d/5_dense18/BiasAdd/ReadVariableOp?
ensemble4d/5_dense18/BiasAddBiasAdd%ensemble4d/5_dense18/MatMul:product:03ensemble4d/5_dense18/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
ensemble4d/5_dense18/BiasAdd?
ensemble4d/5_dense18/LeakyRelu	LeakyRelu%ensemble4d/5_dense18/BiasAdd:output:0*'
_output_shapes
:?????????2 
ensemble4d/5_dense18/LeakyRelu?
ensemble4d/dropout_1/IdentityIdentity*ensemble4d/dense_1/LeakyRelu:activations:0*
T0*(
_output_shapes
:??????????2
ensemble4d/dropout_1/Identity?
$ensemble4d/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2&
$ensemble4d/concatenate_1/concat/axis?
ensemble4d/concatenate_1/concatConcatV2,ensemble4d/5_dense18/LeakyRelu:activations:0&ensemble4d/dropout_1/Identity:output:0-ensemble4d/concatenate_1/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2!
ensemble4d/concatenate_1/concat?
/ensemble4d/combined_input/MatMul/ReadVariableOpReadVariableOp8ensemble4d_combined_input_matmul_readvariableop_resource*
_output_shapes
:	?	*
dtype021
/ensemble4d/combined_input/MatMul/ReadVariableOp?
 ensemble4d/combined_input/MatMulMatMul(ensemble4d/concatenate_1/concat:output:07ensemble4d/combined_input/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2"
 ensemble4d/combined_input/MatMul?
0ensemble4d/combined_input/BiasAdd/ReadVariableOpReadVariableOp9ensemble4d_combined_input_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype022
0ensemble4d/combined_input/BiasAdd/ReadVariableOp?
!ensemble4d/combined_input/BiasAddBiasAdd*ensemble4d/combined_input/MatMul:product:08ensemble4d/combined_input/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2#
!ensemble4d/combined_input/BiasAdd?
#ensemble4d/combined_input/LeakyRelu	LeakyRelu*ensemble4d/combined_input/BiasAdd:output:0*'
_output_shapes
:?????????	2%
#ensemble4d/combined_input/LeakyRelu?
0ensemble4d/ensemble_output/MatMul/ReadVariableOpReadVariableOp9ensemble4d_ensemble_output_matmul_readvariableop_resource*
_output_shapes

:	*
dtype022
0ensemble4d/ensemble_output/MatMul/ReadVariableOp?
!ensemble4d/ensemble_output/MatMulMatMul1ensemble4d/combined_input/LeakyRelu:activations:08ensemble4d/ensemble_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2#
!ensemble4d/ensemble_output/MatMul?
1ensemble4d/ensemble_output/BiasAdd/ReadVariableOpReadVariableOp:ensemble4d_ensemble_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype023
1ensemble4d/ensemble_output/BiasAdd/ReadVariableOp?
"ensemble4d/ensemble_output/BiasAddBiasAdd+ensemble4d/ensemble_output/MatMul:product:09ensemble4d/ensemble_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2$
"ensemble4d/ensemble_output/BiasAdd?
"ensemble4d/ensemble_output/SigmoidSigmoid+ensemble4d/ensemble_output/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2$
"ensemble4d/ensemble_output/Sigmoid?
IdentityIdentity&ensemble4d/ensemble_output/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp,^ensemble4d/1_dense18/BiasAdd/ReadVariableOp+^ensemble4d/1_dense18/MatMul/ReadVariableOp,^ensemble4d/2_dense32/BiasAdd/ReadVariableOp+^ensemble4d/2_dense32/MatMul/ReadVariableOp,^ensemble4d/3_dense64/BiasAdd/ReadVariableOp+^ensemble4d/3_dense64/MatMul/ReadVariableOp,^ensemble4d/4_dense32/BiasAdd/ReadVariableOp+^ensemble4d/4_dense32/MatMul/ReadVariableOp,^ensemble4d/5_dense18/BiasAdd/ReadVariableOp+^ensemble4d/5_dense18/MatMul/ReadVariableOpA^ensemble4d/batch_normalization_5/FusedBatchNormV3/ReadVariableOpC^ensemble4d/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_10^ensemble4d/batch_normalization_5/ReadVariableOp2^ensemble4d/batch_normalization_5/ReadVariableOp_1A^ensemble4d/batch_normalization_6/FusedBatchNormV3/ReadVariableOpC^ensemble4d/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_10^ensemble4d/batch_normalization_6/ReadVariableOp2^ensemble4d/batch_normalization_6/ReadVariableOp_1A^ensemble4d/batch_normalization_7/FusedBatchNormV3/ReadVariableOpC^ensemble4d/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_10^ensemble4d/batch_normalization_7/ReadVariableOp2^ensemble4d/batch_normalization_7/ReadVariableOp_1A^ensemble4d/batch_normalization_8/FusedBatchNormV3/ReadVariableOpC^ensemble4d/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_10^ensemble4d/batch_normalization_8/ReadVariableOp2^ensemble4d/batch_normalization_8/ReadVariableOp_1A^ensemble4d/batch_normalization_9/FusedBatchNormV3/ReadVariableOpC^ensemble4d/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_10^ensemble4d/batch_normalization_9/ReadVariableOp2^ensemble4d/batch_normalization_9/ReadVariableOp_11^ensemble4d/combined_input/BiasAdd/ReadVariableOp0^ensemble4d/combined_input/MatMul/ReadVariableOp+^ensemble4d/conv3d_5/BiasAdd/ReadVariableOp*^ensemble4d/conv3d_5/Conv3D/ReadVariableOp+^ensemble4d/conv3d_6/BiasAdd/ReadVariableOp*^ensemble4d/conv3d_6/Conv3D/ReadVariableOp+^ensemble4d/conv3d_7/BiasAdd/ReadVariableOp*^ensemble4d/conv3d_7/Conv3D/ReadVariableOp+^ensemble4d/conv3d_8/BiasAdd/ReadVariableOp*^ensemble4d/conv3d_8/Conv3D/ReadVariableOp+^ensemble4d/conv3d_9/BiasAdd/ReadVariableOp*^ensemble4d/conv3d_9/Conv3D/ReadVariableOp*^ensemble4d/dense_1/BiasAdd/ReadVariableOp)^ensemble4d/dense_1/MatMul/ReadVariableOp2^ensemble4d/ensemble_output/BiasAdd/ReadVariableOp1^ensemble4d/ensemble_output/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????
:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+ensemble4d/1_dense18/BiasAdd/ReadVariableOp+ensemble4d/1_dense18/BiasAdd/ReadVariableOp2X
*ensemble4d/1_dense18/MatMul/ReadVariableOp*ensemble4d/1_dense18/MatMul/ReadVariableOp2Z
+ensemble4d/2_dense32/BiasAdd/ReadVariableOp+ensemble4d/2_dense32/BiasAdd/ReadVariableOp2X
*ensemble4d/2_dense32/MatMul/ReadVariableOp*ensemble4d/2_dense32/MatMul/ReadVariableOp2Z
+ensemble4d/3_dense64/BiasAdd/ReadVariableOp+ensemble4d/3_dense64/BiasAdd/ReadVariableOp2X
*ensemble4d/3_dense64/MatMul/ReadVariableOp*ensemble4d/3_dense64/MatMul/ReadVariableOp2Z
+ensemble4d/4_dense32/BiasAdd/ReadVariableOp+ensemble4d/4_dense32/BiasAdd/ReadVariableOp2X
*ensemble4d/4_dense32/MatMul/ReadVariableOp*ensemble4d/4_dense32/MatMul/ReadVariableOp2Z
+ensemble4d/5_dense18/BiasAdd/ReadVariableOp+ensemble4d/5_dense18/BiasAdd/ReadVariableOp2X
*ensemble4d/5_dense18/MatMul/ReadVariableOp*ensemble4d/5_dense18/MatMul/ReadVariableOp2?
@ensemble4d/batch_normalization_5/FusedBatchNormV3/ReadVariableOp@ensemble4d/batch_normalization_5/FusedBatchNormV3/ReadVariableOp2?
Bensemble4d/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1Bensemble4d/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12b
/ensemble4d/batch_normalization_5/ReadVariableOp/ensemble4d/batch_normalization_5/ReadVariableOp2f
1ensemble4d/batch_normalization_5/ReadVariableOp_11ensemble4d/batch_normalization_5/ReadVariableOp_12?
@ensemble4d/batch_normalization_6/FusedBatchNormV3/ReadVariableOp@ensemble4d/batch_normalization_6/FusedBatchNormV3/ReadVariableOp2?
Bensemble4d/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1Bensemble4d/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12b
/ensemble4d/batch_normalization_6/ReadVariableOp/ensemble4d/batch_normalization_6/ReadVariableOp2f
1ensemble4d/batch_normalization_6/ReadVariableOp_11ensemble4d/batch_normalization_6/ReadVariableOp_12?
@ensemble4d/batch_normalization_7/FusedBatchNormV3/ReadVariableOp@ensemble4d/batch_normalization_7/FusedBatchNormV3/ReadVariableOp2?
Bensemble4d/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1Bensemble4d/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12b
/ensemble4d/batch_normalization_7/ReadVariableOp/ensemble4d/batch_normalization_7/ReadVariableOp2f
1ensemble4d/batch_normalization_7/ReadVariableOp_11ensemble4d/batch_normalization_7/ReadVariableOp_12?
@ensemble4d/batch_normalization_8/FusedBatchNormV3/ReadVariableOp@ensemble4d/batch_normalization_8/FusedBatchNormV3/ReadVariableOp2?
Bensemble4d/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1Bensemble4d/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12b
/ensemble4d/batch_normalization_8/ReadVariableOp/ensemble4d/batch_normalization_8/ReadVariableOp2f
1ensemble4d/batch_normalization_8/ReadVariableOp_11ensemble4d/batch_normalization_8/ReadVariableOp_12?
@ensemble4d/batch_normalization_9/FusedBatchNormV3/ReadVariableOp@ensemble4d/batch_normalization_9/FusedBatchNormV3/ReadVariableOp2?
Bensemble4d/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1Bensemble4d/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12b
/ensemble4d/batch_normalization_9/ReadVariableOp/ensemble4d/batch_normalization_9/ReadVariableOp2f
1ensemble4d/batch_normalization_9/ReadVariableOp_11ensemble4d/batch_normalization_9/ReadVariableOp_12d
0ensemble4d/combined_input/BiasAdd/ReadVariableOp0ensemble4d/combined_input/BiasAdd/ReadVariableOp2b
/ensemble4d/combined_input/MatMul/ReadVariableOp/ensemble4d/combined_input/MatMul/ReadVariableOp2X
*ensemble4d/conv3d_5/BiasAdd/ReadVariableOp*ensemble4d/conv3d_5/BiasAdd/ReadVariableOp2V
)ensemble4d/conv3d_5/Conv3D/ReadVariableOp)ensemble4d/conv3d_5/Conv3D/ReadVariableOp2X
*ensemble4d/conv3d_6/BiasAdd/ReadVariableOp*ensemble4d/conv3d_6/BiasAdd/ReadVariableOp2V
)ensemble4d/conv3d_6/Conv3D/ReadVariableOp)ensemble4d/conv3d_6/Conv3D/ReadVariableOp2X
*ensemble4d/conv3d_7/BiasAdd/ReadVariableOp*ensemble4d/conv3d_7/BiasAdd/ReadVariableOp2V
)ensemble4d/conv3d_7/Conv3D/ReadVariableOp)ensemble4d/conv3d_7/Conv3D/ReadVariableOp2X
*ensemble4d/conv3d_8/BiasAdd/ReadVariableOp*ensemble4d/conv3d_8/BiasAdd/ReadVariableOp2V
)ensemble4d/conv3d_8/Conv3D/ReadVariableOp)ensemble4d/conv3d_8/Conv3D/ReadVariableOp2X
*ensemble4d/conv3d_9/BiasAdd/ReadVariableOp*ensemble4d/conv3d_9/BiasAdd/ReadVariableOp2V
)ensemble4d/conv3d_9/Conv3D/ReadVariableOp)ensemble4d/conv3d_9/Conv3D/ReadVariableOp2V
)ensemble4d/dense_1/BiasAdd/ReadVariableOp)ensemble4d/dense_1/BiasAdd/ReadVariableOp2T
(ensemble4d/dense_1/MatMul/ReadVariableOp(ensemble4d/dense_1/MatMul/ReadVariableOp2f
1ensemble4d/ensemble_output/BiasAdd/ReadVariableOp1ensemble4d/ensemble_output/BiasAdd/ReadVariableOp2d
0ensemble4d/ensemble_output/MatMul/ReadVariableOp0ensemble4d/ensemble_output/MatMul/ReadVariableOp:S O
'
_output_shapes
:?????????

$
_user_specified_name
svm_inputs:c_
5
_output_shapes#
!:???????????
&
_user_specified_nameimg3d_inputs
?
?
+__inference_3_dense64_layer_call_fn_9258450

inputs
unknown: @
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_3_dense64_layer_call_and_return_conditional_losses_92556462
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?	
?
7__inference_batch_normalization_7_layer_call_fn_9257962

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8????????????????????????????????????@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_92549112
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*N
_output_shapes<
::8????????????????????????????????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8????????????????????????????????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8????????????????????????????????????@
 
_user_specified_nameinputs
?
?
+__inference_4_dense32_layer_call_fn_9258492

inputs
unknown:@ 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_4_dense32_layer_call_and_return_conditional_losses_92556802
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
??
?'
G__inference_ensemble4d_layer_call_and_return_conditional_losses_9257398
inputs_0
inputs_1E
'conv3d_5_conv3d_readvariableop_resource: 6
(conv3d_5_biasadd_readvariableop_resource: ;
-batch_normalization_5_readvariableop_resource: =
/batch_normalization_5_readvariableop_1_resource: L
>batch_normalization_5_fusedbatchnormv3_readvariableop_resource: N
@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource: E
'conv3d_6_conv3d_readvariableop_resource:  6
(conv3d_6_biasadd_readvariableop_resource: ;
-batch_normalization_6_readvariableop_resource: =
/batch_normalization_6_readvariableop_1_resource: L
>batch_normalization_6_fusedbatchnormv3_readvariableop_resource: N
@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource: E
'conv3d_7_conv3d_readvariableop_resource: @6
(conv3d_7_biasadd_readvariableop_resource:@;
-batch_normalization_7_readvariableop_resource:@=
/batch_normalization_7_readvariableop_1_resource:@L
>batch_normalization_7_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource:@F
'conv3d_8_conv3d_readvariableop_resource:@?7
(conv3d_8_biasadd_readvariableop_resource:	?<
-batch_normalization_8_readvariableop_resource:	?>
/batch_normalization_8_readvariableop_1_resource:	?M
>batch_normalization_8_fusedbatchnormv3_readvariableop_resource:	?O
@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource:	?G
'conv3d_9_conv3d_readvariableop_resource:??7
(conv3d_9_biasadd_readvariableop_resource:	?8
&dense18_matmul_readvariableop_resource:
5
'dense18_biasadd_readvariableop_resource:<
-batch_normalization_9_readvariableop_resource:	?>
/batch_normalization_9_readvariableop_1_resource:	?M
>batch_normalization_9_fusedbatchnormv3_readvariableop_resource:	?O
@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource:	?8
&dense32_matmul_readvariableop_resource: 5
'dense32_biasadd_readvariableop_resource: 8
&dense64_matmul_readvariableop_resource: @5
'dense64_biasadd_readvariableop_resource:@:
&dense_1_matmul_readvariableop_resource:
??6
'dense_1_biasadd_readvariableop_resource:	?:
(dense32_matmul_readvariableop_resource_0:@ 7
)dense32_biasadd_readvariableop_resource_0: :
(dense18_matmul_readvariableop_resource_0: 7
)dense18_biasadd_readvariableop_resource_0:@
-combined_input_matmul_readvariableop_resource:	?	<
.combined_input_biasadd_readvariableop_resource:	@
.ensemble_output_matmul_readvariableop_resource:	=
/ensemble_output_biasadd_readvariableop_resource:
identity?? 1_dense18/BiasAdd/ReadVariableOp?1_dense18/MatMul/ReadVariableOp? 2_dense32/BiasAdd/ReadVariableOp?2_dense32/MatMul/ReadVariableOp? 3_dense64/BiasAdd/ReadVariableOp?3_dense64/MatMul/ReadVariableOp? 4_dense32/BiasAdd/ReadVariableOp?4_dense32/MatMul/ReadVariableOp? 5_dense18/BiasAdd/ReadVariableOp?5_dense18/MatMul/ReadVariableOp?5batch_normalization_5/FusedBatchNormV3/ReadVariableOp?7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_5/ReadVariableOp?&batch_normalization_5/ReadVariableOp_1?5batch_normalization_6/FusedBatchNormV3/ReadVariableOp?7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_6/ReadVariableOp?&batch_normalization_6/ReadVariableOp_1?5batch_normalization_7/FusedBatchNormV3/ReadVariableOp?7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_7/ReadVariableOp?&batch_normalization_7/ReadVariableOp_1?5batch_normalization_8/FusedBatchNormV3/ReadVariableOp?7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_8/ReadVariableOp?&batch_normalization_8/ReadVariableOp_1?5batch_normalization_9/FusedBatchNormV3/ReadVariableOp?7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_9/ReadVariableOp?&batch_normalization_9/ReadVariableOp_1?%combined_input/BiasAdd/ReadVariableOp?$combined_input/MatMul/ReadVariableOp?conv3d_5/BiasAdd/ReadVariableOp?conv3d_5/Conv3D/ReadVariableOp?conv3d_6/BiasAdd/ReadVariableOp?conv3d_6/Conv3D/ReadVariableOp?conv3d_7/BiasAdd/ReadVariableOp?conv3d_7/Conv3D/ReadVariableOp?conv3d_8/BiasAdd/ReadVariableOp?conv3d_8/Conv3D/ReadVariableOp?conv3d_9/BiasAdd/ReadVariableOp?conv3d_9/Conv3D/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?&ensemble_output/BiasAdd/ReadVariableOp?%ensemble_output/MatMul/ReadVariableOp?
conv3d_5/Conv3D/ReadVariableOpReadVariableOp'conv3d_5_conv3d_readvariableop_resource**
_output_shapes
: *
dtype02 
conv3d_5/Conv3D/ReadVariableOp?
conv3d_5/Conv3DConv3Dinputs_1&conv3d_5/Conv3D/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:??????????? *
paddingSAME*
strides	
2
conv3d_5/Conv3D?
conv3d_5/BiasAdd/ReadVariableOpReadVariableOp(conv3d_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv3d_5/BiasAdd/ReadVariableOp?
conv3d_5/BiasAddBiasAddconv3d_5/Conv3D:output:0'conv3d_5/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:??????????? 2
conv3d_5/BiasAdd?
conv3d_5/LeakyRelu	LeakyReluconv3d_5/BiasAdd:output:0*5
_output_shapes#
!:??????????? 2
conv3d_5/LeakyRelu?
max_pooling3d_5/MaxPool3D	MaxPool3D conv3d_5/LeakyRelu:activations:0*
T0*3
_output_shapes!
:?????????@@ *
ksize	
*
paddingVALID*
strides	
2
max_pooling3d_5/MaxPool3D?
$batch_normalization_5/ReadVariableOpReadVariableOp-batch_normalization_5_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_5/ReadVariableOp?
&batch_normalization_5/ReadVariableOp_1ReadVariableOp/batch_normalization_5_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_5/ReadVariableOp_1?
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3"max_pooling3d_5/MaxPool3D:output:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*O
_output_shapes=
;:?????????@@ : : : : :*
data_formatNDHWC*
epsilon%o?:*
is_training( 2(
&batch_normalization_5/FusedBatchNormV3?
conv3d_6/Conv3D/ReadVariableOpReadVariableOp'conv3d_6_conv3d_readvariableop_resource**
_output_shapes
:  *
dtype02 
conv3d_6/Conv3D/ReadVariableOp?
conv3d_6/Conv3DConv3D*batch_normalization_5/FusedBatchNormV3:y:0&conv3d_6/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@@ *
paddingSAME*
strides	
2
conv3d_6/Conv3D?
conv3d_6/BiasAdd/ReadVariableOpReadVariableOp(conv3d_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv3d_6/BiasAdd/ReadVariableOp?
conv3d_6/BiasAddBiasAddconv3d_6/Conv3D:output:0'conv3d_6/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@@ 2
conv3d_6/BiasAdd?
conv3d_6/LeakyRelu	LeakyReluconv3d_6/BiasAdd:output:0*3
_output_shapes!
:?????????@@ 2
conv3d_6/LeakyRelu?
max_pooling3d_6/MaxPool3D	MaxPool3D conv3d_6/LeakyRelu:activations:0*
T0*3
_output_shapes!
:?????????@@ *
ksize	
*
paddingVALID*
strides	
2
max_pooling3d_6/MaxPool3D?
$batch_normalization_6/ReadVariableOpReadVariableOp-batch_normalization_6_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_6/ReadVariableOp?
&batch_normalization_6/ReadVariableOp_1ReadVariableOp/batch_normalization_6_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_6/ReadVariableOp_1?
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_6/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_6/FusedBatchNormV3FusedBatchNormV3"max_pooling3d_6/MaxPool3D:output:0,batch_normalization_6/ReadVariableOp:value:0.batch_normalization_6/ReadVariableOp_1:value:0=batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*O
_output_shapes=
;:?????????@@ : : : : :*
data_formatNDHWC*
epsilon%o?:*
is_training( 2(
&batch_normalization_6/FusedBatchNormV3?
conv3d_7/Conv3D/ReadVariableOpReadVariableOp'conv3d_7_conv3d_readvariableop_resource**
_output_shapes
: @*
dtype02 
conv3d_7/Conv3D/ReadVariableOp?
conv3d_7/Conv3DConv3D*batch_normalization_6/FusedBatchNormV3:y:0&conv3d_7/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@@@*
paddingSAME*
strides	
2
conv3d_7/Conv3D?
conv3d_7/BiasAdd/ReadVariableOpReadVariableOp(conv3d_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv3d_7/BiasAdd/ReadVariableOp?
conv3d_7/BiasAddBiasAddconv3d_7/Conv3D:output:0'conv3d_7/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@@@2
conv3d_7/BiasAdd?
conv3d_7/LeakyRelu	LeakyReluconv3d_7/BiasAdd:output:0*3
_output_shapes!
:?????????@@@2
conv3d_7/LeakyRelu?
max_pooling3d_7/MaxPool3D	MaxPool3D conv3d_7/LeakyRelu:activations:0*
T0*3
_output_shapes!
:?????????@@@*
ksize	
*
paddingVALID*
strides	
2
max_pooling3d_7/MaxPool3D?
$batch_normalization_7/ReadVariableOpReadVariableOp-batch_normalization_7_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_7/ReadVariableOp?
&batch_normalization_7/ReadVariableOp_1ReadVariableOp/batch_normalization_7_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_7/ReadVariableOp_1?
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_7/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3"max_pooling3d_7/MaxPool3D:output:0,batch_normalization_7/ReadVariableOp:value:0.batch_normalization_7/ReadVariableOp_1:value:0=batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*O
_output_shapes=
;:?????????@@@:@:@:@:@:*
data_formatNDHWC*
epsilon%o?:*
is_training( 2(
&batch_normalization_7/FusedBatchNormV3?
conv3d_8/Conv3D/ReadVariableOpReadVariableOp'conv3d_8_conv3d_readvariableop_resource*+
_output_shapes
:@?*
dtype02 
conv3d_8/Conv3D/ReadVariableOp?
conv3d_8/Conv3DConv3D*batch_normalization_7/FusedBatchNormV3:y:0&conv3d_8/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :?????????@@?*
paddingSAME*
strides	
2
conv3d_8/Conv3D?
conv3d_8/BiasAdd/ReadVariableOpReadVariableOp(conv3d_8_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv3d_8/BiasAdd/ReadVariableOp?
conv3d_8/BiasAddBiasAddconv3d_8/Conv3D:output:0'conv3d_8/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :?????????@@?2
conv3d_8/BiasAdd?
conv3d_8/LeakyRelu	LeakyReluconv3d_8/BiasAdd:output:0*4
_output_shapes"
 :?????????@@?2
conv3d_8/LeakyRelu?
max_pooling3d_8/MaxPool3D	MaxPool3D conv3d_8/LeakyRelu:activations:0*
T0*4
_output_shapes"
 :?????????@@?*
ksize	
*
paddingVALID*
strides	
2
max_pooling3d_8/MaxPool3D?
$batch_normalization_8/ReadVariableOpReadVariableOp-batch_normalization_8_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$batch_normalization_8/ReadVariableOp?
&batch_normalization_8/ReadVariableOp_1ReadVariableOp/batch_normalization_8_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&batch_normalization_8/ReadVariableOp_1?
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype027
5batch_normalization_8/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype029
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_8/FusedBatchNormV3FusedBatchNormV3"max_pooling3d_8/MaxPool3D:output:0,batch_normalization_8/ReadVariableOp:value:0.batch_normalization_8/ReadVariableOp_1:value:0=batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*T
_output_shapesB
@:?????????@@?:?:?:?:?:*
data_formatNDHWC*
epsilon%o?:*
is_training( 2(
&batch_normalization_8/FusedBatchNormV3?
conv3d_9/Conv3D/ReadVariableOpReadVariableOp'conv3d_9_conv3d_readvariableop_resource*,
_output_shapes
:??*
dtype02 
conv3d_9/Conv3D/ReadVariableOp?
conv3d_9/Conv3DConv3D*batch_normalization_8/FusedBatchNormV3:y:0&conv3d_9/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :?????????@@?*
paddingSAME*
strides	
2
conv3d_9/Conv3D?
conv3d_9/BiasAdd/ReadVariableOpReadVariableOp(conv3d_9_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv3d_9/BiasAdd/ReadVariableOp?
conv3d_9/BiasAddBiasAddconv3d_9/Conv3D:output:0'conv3d_9/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :?????????@@?2
conv3d_9/BiasAdd?
conv3d_9/LeakyRelu	LeakyReluconv3d_9/BiasAdd:output:0*4
_output_shapes"
 :?????????@@?2
conv3d_9/LeakyRelu?
max_pooling3d_9/MaxPool3D	MaxPool3D conv3d_9/LeakyRelu:activations:0*
T0*4
_output_shapes"
 :?????????@@?*
ksize	
*
paddingVALID*
strides	
2
max_pooling3d_9/MaxPool3D?
1_dense18/MatMul/ReadVariableOpReadVariableOp&dense18_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02!
1_dense18/MatMul/ReadVariableOp?
1_dense18/MatMulMatMulinputs_0'1_dense18/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
1_dense18/MatMul?
 1_dense18/BiasAdd/ReadVariableOpReadVariableOp'dense18_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 1_dense18/BiasAdd/ReadVariableOp?
1_dense18/BiasAddBiasAdd1_dense18/MatMul:product:0(1_dense18/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
1_dense18/BiasAdd|
1_dense18/LeakyRelu	LeakyRelu1_dense18/BiasAdd:output:0*'
_output_shapes
:?????????2
1_dense18/LeakyRelu?
$batch_normalization_9/ReadVariableOpReadVariableOp-batch_normalization_9_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$batch_normalization_9/ReadVariableOp?
&batch_normalization_9/ReadVariableOp_1ReadVariableOp/batch_normalization_9_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&batch_normalization_9/ReadVariableOp_1?
5batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype027
5batch_normalization_9/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype029
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_9/FusedBatchNormV3FusedBatchNormV3"max_pooling3d_9/MaxPool3D:output:0,batch_normalization_9/ReadVariableOp:value:0.batch_normalization_9/ReadVariableOp_1:value:0=batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*T
_output_shapesB
@:?????????@@?:?:?:?:?:*
data_formatNDHWC*
epsilon%o?:*
is_training( 2(
&batch_normalization_9/FusedBatchNormV3?
2_dense32/MatMul/ReadVariableOpReadVariableOp&dense32_matmul_readvariableop_resource*
_output_shapes

: *
dtype02!
2_dense32/MatMul/ReadVariableOp?
2_dense32/MatMulMatMul!1_dense18/LeakyRelu:activations:0'2_dense32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
2_dense32/MatMul?
 2_dense32/BiasAdd/ReadVariableOpReadVariableOp'dense32_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 2_dense32/BiasAdd/ReadVariableOp?
2_dense32/BiasAddBiasAdd2_dense32/MatMul:product:0(2_dense32/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
2_dense32/BiasAdd|
2_dense32/LeakyRelu	LeakyRelu2_dense32/BiasAdd:output:0*'
_output_shapes
:????????? 2
2_dense32/LeakyRelu?
1global_average_pooling3d_1/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         23
1global_average_pooling3d_1/Mean/reduction_indices?
global_average_pooling3d_1/MeanMean*batch_normalization_9/FusedBatchNormV3:y:0:global_average_pooling3d_1/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2!
global_average_pooling3d_1/Mean?
3_dense64/MatMul/ReadVariableOpReadVariableOp&dense64_matmul_readvariableop_resource*
_output_shapes

: @*
dtype02!
3_dense64/MatMul/ReadVariableOp?
3_dense64/MatMulMatMul!2_dense32/LeakyRelu:activations:0'3_dense64/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
3_dense64/MatMul?
 3_dense64/BiasAdd/ReadVariableOpReadVariableOp'dense64_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 3_dense64/BiasAdd/ReadVariableOp?
3_dense64/BiasAddBiasAdd3_dense64/MatMul:product:0(3_dense64/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
3_dense64/BiasAdd|
3_dense64/LeakyRelu	LeakyRelu3_dense64/BiasAdd:output:0*'
_output_shapes
:?????????@2
3_dense64/LeakyRelu?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMul(global_average_pooling3d_1/Mean:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_1/BiasAddw
dense_1/LeakyRelu	LeakyReludense_1/BiasAdd:output:0*(
_output_shapes
:??????????2
dense_1/LeakyRelu?
4_dense32/MatMul/ReadVariableOpReadVariableOp(dense32_matmul_readvariableop_resource_0*
_output_shapes

:@ *
dtype02!
4_dense32/MatMul/ReadVariableOp?
4_dense32/MatMulMatMul!3_dense64/LeakyRelu:activations:0'4_dense32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
4_dense32/MatMul?
 4_dense32/BiasAdd/ReadVariableOpReadVariableOp)dense32_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype02"
 4_dense32/BiasAdd/ReadVariableOp?
4_dense32/BiasAddBiasAdd4_dense32/MatMul:product:0(4_dense32/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
4_dense32/BiasAdd|
4_dense32/LeakyRelu	LeakyRelu4_dense32/BiasAdd:output:0*'
_output_shapes
:????????? 2
4_dense32/LeakyRelu?
5_dense18/MatMul/ReadVariableOpReadVariableOp(dense18_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype02!
5_dense18/MatMul/ReadVariableOp?
5_dense18/MatMulMatMul!4_dense32/LeakyRelu:activations:0'5_dense18/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
5_dense18/MatMul?
 5_dense18/BiasAdd/ReadVariableOpReadVariableOp)dense18_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype02"
 5_dense18/BiasAdd/ReadVariableOp?
5_dense18/BiasAddBiasAdd5_dense18/MatMul:product:0(5_dense18/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
5_dense18/BiasAdd|
5_dense18/LeakyRelu	LeakyRelu5_dense18/BiasAdd:output:0*'
_output_shapes
:?????????2
5_dense18/LeakyRelu?
dropout_1/IdentityIdentitydense_1/LeakyRelu:activations:0*
T0*(
_output_shapes
:??????????2
dropout_1/Identityx
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_1/concat/axis?
concatenate_1/concatConcatV2!5_dense18/LeakyRelu:activations:0dropout_1/Identity:output:0"concatenate_1/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
concatenate_1/concat?
$combined_input/MatMul/ReadVariableOpReadVariableOp-combined_input_matmul_readvariableop_resource*
_output_shapes
:	?	*
dtype02&
$combined_input/MatMul/ReadVariableOp?
combined_input/MatMulMatMulconcatenate_1/concat:output:0,combined_input/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2
combined_input/MatMul?
%combined_input/BiasAdd/ReadVariableOpReadVariableOp.combined_input_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02'
%combined_input/BiasAdd/ReadVariableOp?
combined_input/BiasAddBiasAddcombined_input/MatMul:product:0-combined_input/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2
combined_input/BiasAdd?
combined_input/LeakyRelu	LeakyRelucombined_input/BiasAdd:output:0*'
_output_shapes
:?????????	2
combined_input/LeakyRelu?
%ensemble_output/MatMul/ReadVariableOpReadVariableOp.ensemble_output_matmul_readvariableop_resource*
_output_shapes

:	*
dtype02'
%ensemble_output/MatMul/ReadVariableOp?
ensemble_output/MatMulMatMul&combined_input/LeakyRelu:activations:0-ensemble_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
ensemble_output/MatMul?
&ensemble_output/BiasAdd/ReadVariableOpReadVariableOp/ensemble_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&ensemble_output/BiasAdd/ReadVariableOp?
ensemble_output/BiasAddBiasAdd ensemble_output/MatMul:product:0.ensemble_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
ensemble_output/BiasAdd?
ensemble_output/SigmoidSigmoid ensemble_output/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
ensemble_output/Sigmoidv
IdentityIdentityensemble_output/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp!^1_dense18/BiasAdd/ReadVariableOp ^1_dense18/MatMul/ReadVariableOp!^2_dense32/BiasAdd/ReadVariableOp ^2_dense32/MatMul/ReadVariableOp!^3_dense64/BiasAdd/ReadVariableOp ^3_dense64/MatMul/ReadVariableOp!^4_dense32/BiasAdd/ReadVariableOp ^4_dense32/MatMul/ReadVariableOp!^5_dense18/BiasAdd/ReadVariableOp ^5_dense18/MatMul/ReadVariableOp6^batch_normalization_5/FusedBatchNormV3/ReadVariableOp8^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_5/ReadVariableOp'^batch_normalization_5/ReadVariableOp_16^batch_normalization_6/FusedBatchNormV3/ReadVariableOp8^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_6/ReadVariableOp'^batch_normalization_6/ReadVariableOp_16^batch_normalization_7/FusedBatchNormV3/ReadVariableOp8^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_7/ReadVariableOp'^batch_normalization_7/ReadVariableOp_16^batch_normalization_8/FusedBatchNormV3/ReadVariableOp8^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_8/ReadVariableOp'^batch_normalization_8/ReadVariableOp_16^batch_normalization_9/FusedBatchNormV3/ReadVariableOp8^batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_9/ReadVariableOp'^batch_normalization_9/ReadVariableOp_1&^combined_input/BiasAdd/ReadVariableOp%^combined_input/MatMul/ReadVariableOp ^conv3d_5/BiasAdd/ReadVariableOp^conv3d_5/Conv3D/ReadVariableOp ^conv3d_6/BiasAdd/ReadVariableOp^conv3d_6/Conv3D/ReadVariableOp ^conv3d_7/BiasAdd/ReadVariableOp^conv3d_7/Conv3D/ReadVariableOp ^conv3d_8/BiasAdd/ReadVariableOp^conv3d_8/Conv3D/ReadVariableOp ^conv3d_9/BiasAdd/ReadVariableOp^conv3d_9/Conv3D/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp'^ensemble_output/BiasAdd/ReadVariableOp&^ensemble_output/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????
:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2D
 1_dense18/BiasAdd/ReadVariableOp 1_dense18/BiasAdd/ReadVariableOp2B
1_dense18/MatMul/ReadVariableOp1_dense18/MatMul/ReadVariableOp2D
 2_dense32/BiasAdd/ReadVariableOp 2_dense32/BiasAdd/ReadVariableOp2B
2_dense32/MatMul/ReadVariableOp2_dense32/MatMul/ReadVariableOp2D
 3_dense64/BiasAdd/ReadVariableOp 3_dense64/BiasAdd/ReadVariableOp2B
3_dense64/MatMul/ReadVariableOp3_dense64/MatMul/ReadVariableOp2D
 4_dense32/BiasAdd/ReadVariableOp 4_dense32/BiasAdd/ReadVariableOp2B
4_dense32/MatMul/ReadVariableOp4_dense32/MatMul/ReadVariableOp2D
 5_dense18/BiasAdd/ReadVariableOp 5_dense18/BiasAdd/ReadVariableOp2B
5_dense18/MatMul/ReadVariableOp5_dense18/MatMul/ReadVariableOp2n
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp5batch_normalization_5/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_17batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_5/ReadVariableOp$batch_normalization_5/ReadVariableOp2P
&batch_normalization_5/ReadVariableOp_1&batch_normalization_5/ReadVariableOp_12n
5batch_normalization_6/FusedBatchNormV3/ReadVariableOp5batch_normalization_6/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_17batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_6/ReadVariableOp$batch_normalization_6/ReadVariableOp2P
&batch_normalization_6/ReadVariableOp_1&batch_normalization_6/ReadVariableOp_12n
5batch_normalization_7/FusedBatchNormV3/ReadVariableOp5batch_normalization_7/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_17batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_7/ReadVariableOp$batch_normalization_7/ReadVariableOp2P
&batch_normalization_7/ReadVariableOp_1&batch_normalization_7/ReadVariableOp_12n
5batch_normalization_8/FusedBatchNormV3/ReadVariableOp5batch_normalization_8/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_17batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_8/ReadVariableOp$batch_normalization_8/ReadVariableOp2P
&batch_normalization_8/ReadVariableOp_1&batch_normalization_8/ReadVariableOp_12n
5batch_normalization_9/FusedBatchNormV3/ReadVariableOp5batch_normalization_9/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_17batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_9/ReadVariableOp$batch_normalization_9/ReadVariableOp2P
&batch_normalization_9/ReadVariableOp_1&batch_normalization_9/ReadVariableOp_12N
%combined_input/BiasAdd/ReadVariableOp%combined_input/BiasAdd/ReadVariableOp2L
$combined_input/MatMul/ReadVariableOp$combined_input/MatMul/ReadVariableOp2B
conv3d_5/BiasAdd/ReadVariableOpconv3d_5/BiasAdd/ReadVariableOp2@
conv3d_5/Conv3D/ReadVariableOpconv3d_5/Conv3D/ReadVariableOp2B
conv3d_6/BiasAdd/ReadVariableOpconv3d_6/BiasAdd/ReadVariableOp2@
conv3d_6/Conv3D/ReadVariableOpconv3d_6/Conv3D/ReadVariableOp2B
conv3d_7/BiasAdd/ReadVariableOpconv3d_7/BiasAdd/ReadVariableOp2@
conv3d_7/Conv3D/ReadVariableOpconv3d_7/Conv3D/ReadVariableOp2B
conv3d_8/BiasAdd/ReadVariableOpconv3d_8/BiasAdd/ReadVariableOp2@
conv3d_8/Conv3D/ReadVariableOpconv3d_8/Conv3D/ReadVariableOp2B
conv3d_9/BiasAdd/ReadVariableOpconv3d_9/BiasAdd/ReadVariableOp2@
conv3d_9/Conv3D/ReadVariableOpconv3d_9/Conv3D/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2P
&ensemble_output/BiasAdd/ReadVariableOp&ensemble_output/BiasAdd/ReadVariableOp2N
%ensemble_output/MatMul/ReadVariableOp%ensemble_output/MatMul/ReadVariableOp:Q M
'
_output_shapes
:?????????

"
_user_specified_name
inputs/0:_[
5
_output_shapes#
!:???????????
"
_user_specified_name
inputs/1
?
?
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_9257745

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*O
_output_shapes=
;:?????????@@ : : : : :*
data_formatNDHWC*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1{
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*3
_output_shapes!
:?????????@@ 2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????@@ : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:[ W
3
_output_shapes!
:?????????@@ 
 
_user_specified_nameinputs
?
s
W__inference_global_average_pooling3d_1_layer_call_and_return_conditional_losses_9258483

inputs
identity?
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2
Mean/reduction_indicesp
MeanMeaninputsMean/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
Meanb
IdentityIdentityMean:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????@@?:\ X
4
_output_shapes"
 :?????????@@?
 
_user_specified_nameinputs
?
?
%__inference_signature_wrapper_9257026
img3d_inputs

svm_inputs%
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: '
	unknown_5:  
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: (

unknown_11: @

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@)

unknown_17:@?

unknown_18:	?

unknown_19:	?

unknown_20:	?

unknown_21:	?

unknown_22:	?*

unknown_23:??

unknown_24:	?

unknown_25:


unknown_26:

unknown_27:	?

unknown_28:	?

unknown_29:	?

unknown_30:	?

unknown_31: 

unknown_32: 

unknown_33: @

unknown_34:@

unknown_35:
??

unknown_36:	?

unknown_37:@ 

unknown_38: 

unknown_39: 

unknown_40:

unknown_41:	?	

unknown_42:	

unknown_43:	

unknown_44:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall
svm_inputsimg3d_inputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44*;
Tin4
220*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*P
_read_only_resource_inputs2
0.	
 !"#$%&'()*+,-./*0
config_proto 

CPU

GPU2*0J 8? *+
f&R$
"__inference__wrapped_model_92545712
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????:?????????
: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:c _
5
_output_shapes#
!:???????????
&
_user_specified_nameimg3d_inputs:SO
'
_output_shapes
:?????????

$
_user_specified_name
svm_inputs
?
?
*__inference_conv3d_6_layer_call_fn_9257754

inputs%
unknown:  
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@@ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv3d_6_layer_call_and_return_conditional_losses_92554052
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:?????????@@ 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@@ : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:?????????@@ 
 
_user_specified_nameinputs
?
h
L__inference_max_pooling3d_9_layer_call_and_return_conditional_losses_9258292

inputs
identity?
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????*
ksize	
*
paddingVALID*
strides	
2
	MaxPool3D?
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A?????????????????????????????????????????????: {
W
_output_shapesE
C:A?????????????????????????????????????????????
 
_user_specified_nameinputs
?
M
1__inference_max_pooling3d_9_layer_call_fn_9258287

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????@@?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling3d_9_layer_call_and_return_conditional_losses_92555652
PartitionedCally
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :?????????@@?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????@@?:\ X
4
_output_shapes"
 :?????????@@?
 
_user_specified_nameinputs
?
h
L__inference_max_pooling3d_9_layer_call_and_return_conditional_losses_9258297

inputs
identity?
	MaxPool3D	MaxPool3Dinputs*
T0*4
_output_shapes"
 :?????????@@?*
ksize	
*
paddingVALID*
strides	
2
	MaxPool3Ds
IdentityIdentityMaxPool3D:output:0*
T0*4
_output_shapes"
 :?????????@@?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????@@?:\ X
4
_output_shapes"
 :?????????@@?
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_9257855

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*j
_output_shapesX
V:8???????????????????????????????????? : : : : :*
data_formatNDHWC*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*N
_output_shapes<
::8???????????????????????????????????? 2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8???????????????????????????????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:v r
N
_output_shapes<
::8???????????????????????????????????? 
 
_user_specified_nameinputs
?
?
E__inference_conv3d_7_layer_call_and_return_conditional_losses_9257929

inputs<
conv3d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
: @*
dtype02
Conv3D/ReadVariableOp?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@@@*
paddingSAME*
strides	
2
Conv3D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@@@2	
BiasAddj
	LeakyRelu	LeakyReluBiasAdd:output:0*3
_output_shapes!
:?????????@@@2
	LeakyRelu~
IdentityIdentityLeakyRelu:activations:0^NoOp*
T0*3
_output_shapes!
:?????????@@@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@@ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:?????????@@ 
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_9255534

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*T
_output_shapesB
@:?????????@@?:?:?:?:?:*
data_formatNDHWC*
epsilon%o?:*
is_training( 2
FusedBatchNormV3|
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*4
_output_shapes"
 :?????????@@?2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:?????????@@?: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:\ X
4
_output_shapes"
 :?????????@@?
 
_user_specified_nameinputs
?
h
L__inference_max_pooling3d_7_layer_call_and_return_conditional_losses_9257944

inputs
identity?
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????*
ksize	
*
paddingVALID*
strides	
2
	MaxPool3D?
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A?????????????????????????????????????????????: {
W
_output_shapesE
C:A?????????????????????????????????????????????
 
_user_specified_nameinputs
?
e
F__inference_dropout_1_layer_call_and_return_conditional_losses_9258570

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniformg
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *???>2
dropout/Const_1?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/Const_1:output:0*
T0*(
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
+__inference_1_dense18_layer_call_fn_9258266

inputs
unknown:

	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_1_dense18_layer_call_and_return_conditional_losses_92555782
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
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
?	
?
7__inference_batch_normalization_7_layer_call_fn_9258001

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_92561152
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:?????????@@@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????@@@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:?????????@@@
 
_user_specified_nameinputs
?

?
7__inference_batch_normalization_9_layer_call_fn_9258330

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *O
_output_shapes=
;:9?????????????????????????????????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_92552072
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*O
_output_shapes=
;:9?????????????????????????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:9?????????????????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:w s
O
_output_shapes=
;:9?????????????????????????????????????
 
_user_specified_nameinputs
?
X
<__inference_global_average_pooling3d_1_layer_call_fn_9258466

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *`
f[RY
W__inference_global_average_pooling3d_1_layer_call_and_return_conditional_losses_92553212
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A?????????????????????????????????????????????: {
W
_output_shapesE
C:A?????????????????????????????????????????????
 
_user_specified_nameinputs
??
?S
#__inference__traced_restore_9259413
file_prefix>
 assignvariableop_conv3d_5_kernel: .
 assignvariableop_1_conv3d_5_bias: <
.assignvariableop_2_batch_normalization_5_gamma: ;
-assignvariableop_3_batch_normalization_5_beta: B
4assignvariableop_4_batch_normalization_5_moving_mean: F
8assignvariableop_5_batch_normalization_5_moving_variance: @
"assignvariableop_6_conv3d_6_kernel:  .
 assignvariableop_7_conv3d_6_bias: <
.assignvariableop_8_batch_normalization_6_gamma: ;
-assignvariableop_9_batch_normalization_6_beta: C
5assignvariableop_10_batch_normalization_6_moving_mean: G
9assignvariableop_11_batch_normalization_6_moving_variance: A
#assignvariableop_12_conv3d_7_kernel: @/
!assignvariableop_13_conv3d_7_bias:@=
/assignvariableop_14_batch_normalization_7_gamma:@<
.assignvariableop_15_batch_normalization_7_beta:@C
5assignvariableop_16_batch_normalization_7_moving_mean:@G
9assignvariableop_17_batch_normalization_7_moving_variance:@B
#assignvariableop_18_conv3d_8_kernel:@?0
!assignvariableop_19_conv3d_8_bias:	?>
/assignvariableop_20_batch_normalization_8_gamma:	?=
.assignvariableop_21_batch_normalization_8_beta:	?D
5assignvariableop_22_batch_normalization_8_moving_mean:	?H
9assignvariableop_23_batch_normalization_8_moving_variance:	?C
#assignvariableop_24_conv3d_9_kernel:??0
!assignvariableop_25_conv3d_9_bias:	?6
$assignvariableop_26_1_dense18_kernel:
0
"assignvariableop_27_1_dense18_bias:6
$assignvariableop_28_2_dense32_kernel: 0
"assignvariableop_29_2_dense32_bias: >
/assignvariableop_30_batch_normalization_9_gamma:	?=
.assignvariableop_31_batch_normalization_9_beta:	?D
5assignvariableop_32_batch_normalization_9_moving_mean:	?H
9assignvariableop_33_batch_normalization_9_moving_variance:	?6
$assignvariableop_34_3_dense64_kernel: @0
"assignvariableop_35_3_dense64_bias:@6
$assignvariableop_36_4_dense32_kernel:@ 0
"assignvariableop_37_4_dense32_bias: 6
"assignvariableop_38_dense_1_kernel:
??/
 assignvariableop_39_dense_1_bias:	?6
$assignvariableop_40_5_dense18_kernel: 0
"assignvariableop_41_5_dense18_bias:<
)assignvariableop_42_combined_input_kernel:	?	5
'assignvariableop_43_combined_input_bias:	<
*assignvariableop_44_ensemble_output_kernel:	6
(assignvariableop_45_ensemble_output_bias:'
assignvariableop_46_adam_iter:	 )
assignvariableop_47_adam_beta_1: )
assignvariableop_48_adam_beta_2: (
assignvariableop_49_adam_decay: #
assignvariableop_50_total: #
assignvariableop_51_count: %
assignvariableop_52_total_1: %
assignvariableop_53_count_1: H
*assignvariableop_54_adam_conv3d_5_kernel_m: 6
(assignvariableop_55_adam_conv3d_5_bias_m: D
6assignvariableop_56_adam_batch_normalization_5_gamma_m: C
5assignvariableop_57_adam_batch_normalization_5_beta_m: H
*assignvariableop_58_adam_conv3d_6_kernel_m:  6
(assignvariableop_59_adam_conv3d_6_bias_m: D
6assignvariableop_60_adam_batch_normalization_6_gamma_m: C
5assignvariableop_61_adam_batch_normalization_6_beta_m: H
*assignvariableop_62_adam_conv3d_7_kernel_m: @6
(assignvariableop_63_adam_conv3d_7_bias_m:@D
6assignvariableop_64_adam_batch_normalization_7_gamma_m:@C
5assignvariableop_65_adam_batch_normalization_7_beta_m:@I
*assignvariableop_66_adam_conv3d_8_kernel_m:@?7
(assignvariableop_67_adam_conv3d_8_bias_m:	?E
6assignvariableop_68_adam_batch_normalization_8_gamma_m:	?D
5assignvariableop_69_adam_batch_normalization_8_beta_m:	?J
*assignvariableop_70_adam_conv3d_9_kernel_m:??7
(assignvariableop_71_adam_conv3d_9_bias_m:	?=
+assignvariableop_72_adam_1_dense18_kernel_m:
7
)assignvariableop_73_adam_1_dense18_bias_m:=
+assignvariableop_74_adam_2_dense32_kernel_m: 7
)assignvariableop_75_adam_2_dense32_bias_m: E
6assignvariableop_76_adam_batch_normalization_9_gamma_m:	?D
5assignvariableop_77_adam_batch_normalization_9_beta_m:	?=
+assignvariableop_78_adam_3_dense64_kernel_m: @7
)assignvariableop_79_adam_3_dense64_bias_m:@=
+assignvariableop_80_adam_4_dense32_kernel_m:@ 7
)assignvariableop_81_adam_4_dense32_bias_m: =
)assignvariableop_82_adam_dense_1_kernel_m:
??6
'assignvariableop_83_adam_dense_1_bias_m:	?=
+assignvariableop_84_adam_5_dense18_kernel_m: 7
)assignvariableop_85_adam_5_dense18_bias_m:C
0assignvariableop_86_adam_combined_input_kernel_m:	?	<
.assignvariableop_87_adam_combined_input_bias_m:	C
1assignvariableop_88_adam_ensemble_output_kernel_m:	=
/assignvariableop_89_adam_ensemble_output_bias_m:H
*assignvariableop_90_adam_conv3d_5_kernel_v: 6
(assignvariableop_91_adam_conv3d_5_bias_v: D
6assignvariableop_92_adam_batch_normalization_5_gamma_v: C
5assignvariableop_93_adam_batch_normalization_5_beta_v: H
*assignvariableop_94_adam_conv3d_6_kernel_v:  6
(assignvariableop_95_adam_conv3d_6_bias_v: D
6assignvariableop_96_adam_batch_normalization_6_gamma_v: C
5assignvariableop_97_adam_batch_normalization_6_beta_v: H
*assignvariableop_98_adam_conv3d_7_kernel_v: @6
(assignvariableop_99_adam_conv3d_7_bias_v:@E
7assignvariableop_100_adam_batch_normalization_7_gamma_v:@D
6assignvariableop_101_adam_batch_normalization_7_beta_v:@J
+assignvariableop_102_adam_conv3d_8_kernel_v:@?8
)assignvariableop_103_adam_conv3d_8_bias_v:	?F
7assignvariableop_104_adam_batch_normalization_8_gamma_v:	?E
6assignvariableop_105_adam_batch_normalization_8_beta_v:	?K
+assignvariableop_106_adam_conv3d_9_kernel_v:??8
)assignvariableop_107_adam_conv3d_9_bias_v:	?>
,assignvariableop_108_adam_1_dense18_kernel_v:
8
*assignvariableop_109_adam_1_dense18_bias_v:>
,assignvariableop_110_adam_2_dense32_kernel_v: 8
*assignvariableop_111_adam_2_dense32_bias_v: F
7assignvariableop_112_adam_batch_normalization_9_gamma_v:	?E
6assignvariableop_113_adam_batch_normalization_9_beta_v:	?>
,assignvariableop_114_adam_3_dense64_kernel_v: @8
*assignvariableop_115_adam_3_dense64_bias_v:@>
,assignvariableop_116_adam_4_dense32_kernel_v:@ 8
*assignvariableop_117_adam_4_dense32_bias_v: >
*assignvariableop_118_adam_dense_1_kernel_v:
??7
(assignvariableop_119_adam_dense_1_bias_v:	?>
,assignvariableop_120_adam_5_dense18_kernel_v: 8
*assignvariableop_121_adam_5_dense18_bias_v:D
1assignvariableop_122_adam_combined_input_kernel_v:	?	=
/assignvariableop_123_adam_combined_input_bias_v:	D
2assignvariableop_124_adam_ensemble_output_kernel_v:	>
0assignvariableop_125_adam_ensemble_output_bias_v:
identity_127??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_100?AssignVariableOp_101?AssignVariableOp_102?AssignVariableOp_103?AssignVariableOp_104?AssignVariableOp_105?AssignVariableOp_106?AssignVariableOp_107?AssignVariableOp_108?AssignVariableOp_109?AssignVariableOp_11?AssignVariableOp_110?AssignVariableOp_111?AssignVariableOp_112?AssignVariableOp_113?AssignVariableOp_114?AssignVariableOp_115?AssignVariableOp_116?AssignVariableOp_117?AssignVariableOp_118?AssignVariableOp_119?AssignVariableOp_12?AssignVariableOp_120?AssignVariableOp_121?AssignVariableOp_122?AssignVariableOp_123?AssignVariableOp_124?AssignVariableOp_125?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_71?AssignVariableOp_72?AssignVariableOp_73?AssignVariableOp_74?AssignVariableOp_75?AssignVariableOp_76?AssignVariableOp_77?AssignVariableOp_78?AssignVariableOp_79?AssignVariableOp_8?AssignVariableOp_80?AssignVariableOp_81?AssignVariableOp_82?AssignVariableOp_83?AssignVariableOp_84?AssignVariableOp_85?AssignVariableOp_86?AssignVariableOp_87?AssignVariableOp_88?AssignVariableOp_89?AssignVariableOp_9?AssignVariableOp_90?AssignVariableOp_91?AssignVariableOp_92?AssignVariableOp_93?AssignVariableOp_94?AssignVariableOp_95?AssignVariableOp_96?AssignVariableOp_97?AssignVariableOp_98?AssignVariableOp_99?G
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?F
value?FB?FB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*?
dtypes?
?2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp assignvariableop_conv3d_5_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv3d_5_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp.assignvariableop_2_batch_normalization_5_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp-assignvariableop_3_batch_normalization_5_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp4assignvariableop_4_batch_normalization_5_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp8assignvariableop_5_batch_normalization_5_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv3d_6_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv3d_6_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp.assignvariableop_8_batch_normalization_6_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp-assignvariableop_9_batch_normalization_6_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp5assignvariableop_10_batch_normalization_6_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp9assignvariableop_11_batch_normalization_6_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp#assignvariableop_12_conv3d_7_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp!assignvariableop_13_conv3d_7_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp/assignvariableop_14_batch_normalization_7_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp.assignvariableop_15_batch_normalization_7_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp5assignvariableop_16_batch_normalization_7_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp9assignvariableop_17_batch_normalization_7_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp#assignvariableop_18_conv3d_8_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp!assignvariableop_19_conv3d_8_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp/assignvariableop_20_batch_normalization_8_gammaIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp.assignvariableop_21_batch_normalization_8_betaIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp5assignvariableop_22_batch_normalization_8_moving_meanIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp9assignvariableop_23_batch_normalization_8_moving_varianceIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp#assignvariableop_24_conv3d_9_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp!assignvariableop_25_conv3d_9_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp$assignvariableop_26_1_dense18_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp"assignvariableop_27_1_dense18_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp$assignvariableop_28_2_dense32_kernelIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp"assignvariableop_29_2_dense32_biasIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp/assignvariableop_30_batch_normalization_9_gammaIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp.assignvariableop_31_batch_normalization_9_betaIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp5assignvariableop_32_batch_normalization_9_moving_meanIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp9assignvariableop_33_batch_normalization_9_moving_varianceIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp$assignvariableop_34_3_dense64_kernelIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp"assignvariableop_35_3_dense64_biasIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp$assignvariableop_36_4_dense32_kernelIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp"assignvariableop_37_4_dense32_biasIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp"assignvariableop_38_dense_1_kernelIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp assignvariableop_39_dense_1_biasIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp$assignvariableop_40_5_dense18_kernelIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp"assignvariableop_41_5_dense18_biasIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp)assignvariableop_42_combined_input_kernelIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp'assignvariableop_43_combined_input_biasIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp*assignvariableop_44_ensemble_output_kernelIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp(assignvariableop_45_ensemble_output_biasIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOpassignvariableop_46_adam_iterIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOpassignvariableop_47_adam_beta_1Identity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOpassignvariableop_48_adam_beta_2Identity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOpassignvariableop_49_adam_decayIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOpassignvariableop_50_totalIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOpassignvariableop_51_countIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOpassignvariableop_52_total_1Identity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOpassignvariableop_53_count_1Identity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOp*assignvariableop_54_adam_conv3d_5_kernel_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOp(assignvariableop_55_adam_conv3d_5_bias_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOp6assignvariableop_56_adam_batch_normalization_5_gamma_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOp5assignvariableop_57_adam_batch_normalization_5_beta_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOp*assignvariableop_58_adam_conv3d_6_kernel_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59?
AssignVariableOp_59AssignVariableOp(assignvariableop_59_adam_conv3d_6_bias_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60?
AssignVariableOp_60AssignVariableOp6assignvariableop_60_adam_batch_normalization_6_gamma_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61?
AssignVariableOp_61AssignVariableOp5assignvariableop_61_adam_batch_normalization_6_beta_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62?
AssignVariableOp_62AssignVariableOp*assignvariableop_62_adam_conv3d_7_kernel_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63?
AssignVariableOp_63AssignVariableOp(assignvariableop_63_adam_conv3d_7_bias_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64?
AssignVariableOp_64AssignVariableOp6assignvariableop_64_adam_batch_normalization_7_gamma_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65?
AssignVariableOp_65AssignVariableOp5assignvariableop_65_adam_batch_normalization_7_beta_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66?
AssignVariableOp_66AssignVariableOp*assignvariableop_66_adam_conv3d_8_kernel_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67?
AssignVariableOp_67AssignVariableOp(assignvariableop_67_adam_conv3d_8_bias_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68?
AssignVariableOp_68AssignVariableOp6assignvariableop_68_adam_batch_normalization_8_gamma_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69?
AssignVariableOp_69AssignVariableOp5assignvariableop_69_adam_batch_normalization_8_beta_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70?
AssignVariableOp_70AssignVariableOp*assignvariableop_70_adam_conv3d_9_kernel_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71?
AssignVariableOp_71AssignVariableOp(assignvariableop_71_adam_conv3d_9_bias_mIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72?
AssignVariableOp_72AssignVariableOp+assignvariableop_72_adam_1_dense18_kernel_mIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73?
AssignVariableOp_73AssignVariableOp)assignvariableop_73_adam_1_dense18_bias_mIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74?
AssignVariableOp_74AssignVariableOp+assignvariableop_74_adam_2_dense32_kernel_mIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75?
AssignVariableOp_75AssignVariableOp)assignvariableop_75_adam_2_dense32_bias_mIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76?
AssignVariableOp_76AssignVariableOp6assignvariableop_76_adam_batch_normalization_9_gamma_mIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77?
AssignVariableOp_77AssignVariableOp5assignvariableop_77_adam_batch_normalization_9_beta_mIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78?
AssignVariableOp_78AssignVariableOp+assignvariableop_78_adam_3_dense64_kernel_mIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79?
AssignVariableOp_79AssignVariableOp)assignvariableop_79_adam_3_dense64_bias_mIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79n
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:2
Identity_80?
AssignVariableOp_80AssignVariableOp+assignvariableop_80_adam_4_dense32_kernel_mIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_80n
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:2
Identity_81?
AssignVariableOp_81AssignVariableOp)assignvariableop_81_adam_4_dense32_bias_mIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_81n
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:2
Identity_82?
AssignVariableOp_82AssignVariableOp)assignvariableop_82_adam_dense_1_kernel_mIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_82n
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:2
Identity_83?
AssignVariableOp_83AssignVariableOp'assignvariableop_83_adam_dense_1_bias_mIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_83n
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:2
Identity_84?
AssignVariableOp_84AssignVariableOp+assignvariableop_84_adam_5_dense18_kernel_mIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_84n
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:2
Identity_85?
AssignVariableOp_85AssignVariableOp)assignvariableop_85_adam_5_dense18_bias_mIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_85n
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:2
Identity_86?
AssignVariableOp_86AssignVariableOp0assignvariableop_86_adam_combined_input_kernel_mIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_86n
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:2
Identity_87?
AssignVariableOp_87AssignVariableOp.assignvariableop_87_adam_combined_input_bias_mIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_87n
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:2
Identity_88?
AssignVariableOp_88AssignVariableOp1assignvariableop_88_adam_ensemble_output_kernel_mIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_88n
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:2
Identity_89?
AssignVariableOp_89AssignVariableOp/assignvariableop_89_adam_ensemble_output_bias_mIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_89n
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:2
Identity_90?
AssignVariableOp_90AssignVariableOp*assignvariableop_90_adam_conv3d_5_kernel_vIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_90n
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:2
Identity_91?
AssignVariableOp_91AssignVariableOp(assignvariableop_91_adam_conv3d_5_bias_vIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_91n
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:2
Identity_92?
AssignVariableOp_92AssignVariableOp6assignvariableop_92_adam_batch_normalization_5_gamma_vIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_92n
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:2
Identity_93?
AssignVariableOp_93AssignVariableOp5assignvariableop_93_adam_batch_normalization_5_beta_vIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_93n
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:2
Identity_94?
AssignVariableOp_94AssignVariableOp*assignvariableop_94_adam_conv3d_6_kernel_vIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_94n
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:2
Identity_95?
AssignVariableOp_95AssignVariableOp(assignvariableop_95_adam_conv3d_6_bias_vIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_95n
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:2
Identity_96?
AssignVariableOp_96AssignVariableOp6assignvariableop_96_adam_batch_normalization_6_gamma_vIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_96n
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:2
Identity_97?
AssignVariableOp_97AssignVariableOp5assignvariableop_97_adam_batch_normalization_6_beta_vIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_97n
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:2
Identity_98?
AssignVariableOp_98AssignVariableOp*assignvariableop_98_adam_conv3d_7_kernel_vIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_98n
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:2
Identity_99?
AssignVariableOp_99AssignVariableOp(assignvariableop_99_adam_conv3d_7_bias_vIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_99q
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:2
Identity_100?
AssignVariableOp_100AssignVariableOp7assignvariableop_100_adam_batch_normalization_7_gamma_vIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_100q
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:2
Identity_101?
AssignVariableOp_101AssignVariableOp6assignvariableop_101_adam_batch_normalization_7_beta_vIdentity_101:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_101q
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:2
Identity_102?
AssignVariableOp_102AssignVariableOp+assignvariableop_102_adam_conv3d_8_kernel_vIdentity_102:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_102q
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:2
Identity_103?
AssignVariableOp_103AssignVariableOp)assignvariableop_103_adam_conv3d_8_bias_vIdentity_103:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_103q
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:2
Identity_104?
AssignVariableOp_104AssignVariableOp7assignvariableop_104_adam_batch_normalization_8_gamma_vIdentity_104:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_104q
Identity_105IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:2
Identity_105?
AssignVariableOp_105AssignVariableOp6assignvariableop_105_adam_batch_normalization_8_beta_vIdentity_105:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_105q
Identity_106IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:2
Identity_106?
AssignVariableOp_106AssignVariableOp+assignvariableop_106_adam_conv3d_9_kernel_vIdentity_106:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_106q
Identity_107IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
:2
Identity_107?
AssignVariableOp_107AssignVariableOp)assignvariableop_107_adam_conv3d_9_bias_vIdentity_107:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_107q
Identity_108IdentityRestoreV2:tensors:108"/device:CPU:0*
T0*
_output_shapes
:2
Identity_108?
AssignVariableOp_108AssignVariableOp,assignvariableop_108_adam_1_dense18_kernel_vIdentity_108:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_108q
Identity_109IdentityRestoreV2:tensors:109"/device:CPU:0*
T0*
_output_shapes
:2
Identity_109?
AssignVariableOp_109AssignVariableOp*assignvariableop_109_adam_1_dense18_bias_vIdentity_109:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_109q
Identity_110IdentityRestoreV2:tensors:110"/device:CPU:0*
T0*
_output_shapes
:2
Identity_110?
AssignVariableOp_110AssignVariableOp,assignvariableop_110_adam_2_dense32_kernel_vIdentity_110:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_110q
Identity_111IdentityRestoreV2:tensors:111"/device:CPU:0*
T0*
_output_shapes
:2
Identity_111?
AssignVariableOp_111AssignVariableOp*assignvariableop_111_adam_2_dense32_bias_vIdentity_111:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_111q
Identity_112IdentityRestoreV2:tensors:112"/device:CPU:0*
T0*
_output_shapes
:2
Identity_112?
AssignVariableOp_112AssignVariableOp7assignvariableop_112_adam_batch_normalization_9_gamma_vIdentity_112:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_112q
Identity_113IdentityRestoreV2:tensors:113"/device:CPU:0*
T0*
_output_shapes
:2
Identity_113?
AssignVariableOp_113AssignVariableOp6assignvariableop_113_adam_batch_normalization_9_beta_vIdentity_113:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_113q
Identity_114IdentityRestoreV2:tensors:114"/device:CPU:0*
T0*
_output_shapes
:2
Identity_114?
AssignVariableOp_114AssignVariableOp,assignvariableop_114_adam_3_dense64_kernel_vIdentity_114:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_114q
Identity_115IdentityRestoreV2:tensors:115"/device:CPU:0*
T0*
_output_shapes
:2
Identity_115?
AssignVariableOp_115AssignVariableOp*assignvariableop_115_adam_3_dense64_bias_vIdentity_115:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_115q
Identity_116IdentityRestoreV2:tensors:116"/device:CPU:0*
T0*
_output_shapes
:2
Identity_116?
AssignVariableOp_116AssignVariableOp,assignvariableop_116_adam_4_dense32_kernel_vIdentity_116:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_116q
Identity_117IdentityRestoreV2:tensors:117"/device:CPU:0*
T0*
_output_shapes
:2
Identity_117?
AssignVariableOp_117AssignVariableOp*assignvariableop_117_adam_4_dense32_bias_vIdentity_117:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_117q
Identity_118IdentityRestoreV2:tensors:118"/device:CPU:0*
T0*
_output_shapes
:2
Identity_118?
AssignVariableOp_118AssignVariableOp*assignvariableop_118_adam_dense_1_kernel_vIdentity_118:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_118q
Identity_119IdentityRestoreV2:tensors:119"/device:CPU:0*
T0*
_output_shapes
:2
Identity_119?
AssignVariableOp_119AssignVariableOp(assignvariableop_119_adam_dense_1_bias_vIdentity_119:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_119q
Identity_120IdentityRestoreV2:tensors:120"/device:CPU:0*
T0*
_output_shapes
:2
Identity_120?
AssignVariableOp_120AssignVariableOp,assignvariableop_120_adam_5_dense18_kernel_vIdentity_120:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_120q
Identity_121IdentityRestoreV2:tensors:121"/device:CPU:0*
T0*
_output_shapes
:2
Identity_121?
AssignVariableOp_121AssignVariableOp*assignvariableop_121_adam_5_dense18_bias_vIdentity_121:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_121q
Identity_122IdentityRestoreV2:tensors:122"/device:CPU:0*
T0*
_output_shapes
:2
Identity_122?
AssignVariableOp_122AssignVariableOp1assignvariableop_122_adam_combined_input_kernel_vIdentity_122:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_122q
Identity_123IdentityRestoreV2:tensors:123"/device:CPU:0*
T0*
_output_shapes
:2
Identity_123?
AssignVariableOp_123AssignVariableOp/assignvariableop_123_adam_combined_input_bias_vIdentity_123:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_123q
Identity_124IdentityRestoreV2:tensors:124"/device:CPU:0*
T0*
_output_shapes
:2
Identity_124?
AssignVariableOp_124AssignVariableOp2assignvariableop_124_adam_ensemble_output_kernel_vIdentity_124:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_124q
Identity_125IdentityRestoreV2:tensors:125"/device:CPU:0*
T0*
_output_shapes
:2
Identity_125?
AssignVariableOp_125AssignVariableOp0assignvariableop_125_adam_ensemble_output_bias_vIdentity_125:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1259
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_126Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_126i
Identity_127IdentityIdentity_126:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_127?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"%
identity_127Identity_127:output:0*?
_input_shapes?
?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102,
AssignVariableOp_100AssignVariableOp_1002,
AssignVariableOp_101AssignVariableOp_1012,
AssignVariableOp_102AssignVariableOp_1022,
AssignVariableOp_103AssignVariableOp_1032,
AssignVariableOp_104AssignVariableOp_1042,
AssignVariableOp_105AssignVariableOp_1052,
AssignVariableOp_106AssignVariableOp_1062,
AssignVariableOp_107AssignVariableOp_1072,
AssignVariableOp_108AssignVariableOp_1082,
AssignVariableOp_109AssignVariableOp_1092*
AssignVariableOp_11AssignVariableOp_112,
AssignVariableOp_110AssignVariableOp_1102,
AssignVariableOp_111AssignVariableOp_1112,
AssignVariableOp_112AssignVariableOp_1122,
AssignVariableOp_113AssignVariableOp_1132,
AssignVariableOp_114AssignVariableOp_1142,
AssignVariableOp_115AssignVariableOp_1152,
AssignVariableOp_116AssignVariableOp_1162,
AssignVariableOp_117AssignVariableOp_1172,
AssignVariableOp_118AssignVariableOp_1182,
AssignVariableOp_119AssignVariableOp_1192*
AssignVariableOp_12AssignVariableOp_122,
AssignVariableOp_120AssignVariableOp_1202,
AssignVariableOp_121AssignVariableOp_1212,
AssignVariableOp_122AssignVariableOp_1222,
AssignVariableOp_123AssignVariableOp_1232,
AssignVariableOp_124AssignVariableOp_1242,
AssignVariableOp_125AssignVariableOp_1252*
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
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_962*
AssignVariableOp_97AssignVariableOp_972*
AssignVariableOp_98AssignVariableOp_982*
AssignVariableOp_99AssignVariableOp_99:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
L__inference_ensemble_output_layer_call_and_return_conditional_losses_9258623

inputs0
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoidf
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
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
?
v
J__inference_concatenate_1_layer_call_and_return_conditional_losses_9258583
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????:??????????:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/1
?
?
F__inference_1_dense18_layer_call_and_return_conditional_losses_9255578

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd^
	LeakyRelu	LeakyReluBiasAdd:output:0*'
_output_shapes
:?????????2
	LeakyRelur
IdentityIdentityLeakyRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
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
?
s
W__inference_global_average_pooling3d_1_layer_call_and_return_conditional_losses_9255633

inputs
identity?
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2
Mean/reduction_indicesp
MeanMeaninputsMean/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
Meanb
IdentityIdentityMean:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????@@?:\ X
4
_output_shapes"
 :?????????@@?
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_9258423

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*T
_output_shapesB
@:?????????@@?:?:?:?:?:*
data_formatNDHWC*
epsilon%o?:*
is_training( 2
FusedBatchNormV3|
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*4
_output_shapes"
 :?????????@@?2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:?????????@@?: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:\ X
4
_output_shapes"
 :?????????@@?
 
_user_specified_nameinputs
?
?
D__inference_dense_1_layer_call_and_return_conditional_losses_9255663

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd_
	LeakyRelu	LeakyReluBiasAdd:output:0*(
_output_shapes
:??????????2
	LeakyRelus
IdentityIdentityLeakyRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?*
G__inference_ensemble4d_layer_call_and_return_conditional_losses_9257581
inputs_0
inputs_1E
'conv3d_5_conv3d_readvariableop_resource: 6
(conv3d_5_biasadd_readvariableop_resource: ;
-batch_normalization_5_readvariableop_resource: =
/batch_normalization_5_readvariableop_1_resource: L
>batch_normalization_5_fusedbatchnormv3_readvariableop_resource: N
@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource: E
'conv3d_6_conv3d_readvariableop_resource:  6
(conv3d_6_biasadd_readvariableop_resource: ;
-batch_normalization_6_readvariableop_resource: =
/batch_normalization_6_readvariableop_1_resource: L
>batch_normalization_6_fusedbatchnormv3_readvariableop_resource: N
@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource: E
'conv3d_7_conv3d_readvariableop_resource: @6
(conv3d_7_biasadd_readvariableop_resource:@;
-batch_normalization_7_readvariableop_resource:@=
/batch_normalization_7_readvariableop_1_resource:@L
>batch_normalization_7_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource:@F
'conv3d_8_conv3d_readvariableop_resource:@?7
(conv3d_8_biasadd_readvariableop_resource:	?<
-batch_normalization_8_readvariableop_resource:	?>
/batch_normalization_8_readvariableop_1_resource:	?M
>batch_normalization_8_fusedbatchnormv3_readvariableop_resource:	?O
@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource:	?G
'conv3d_9_conv3d_readvariableop_resource:??7
(conv3d_9_biasadd_readvariableop_resource:	?8
&dense18_matmul_readvariableop_resource:
5
'dense18_biasadd_readvariableop_resource:<
-batch_normalization_9_readvariableop_resource:	?>
/batch_normalization_9_readvariableop_1_resource:	?M
>batch_normalization_9_fusedbatchnormv3_readvariableop_resource:	?O
@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource:	?8
&dense32_matmul_readvariableop_resource: 5
'dense32_biasadd_readvariableop_resource: 8
&dense64_matmul_readvariableop_resource: @5
'dense64_biasadd_readvariableop_resource:@:
&dense_1_matmul_readvariableop_resource:
??6
'dense_1_biasadd_readvariableop_resource:	?:
(dense32_matmul_readvariableop_resource_0:@ 7
)dense32_biasadd_readvariableop_resource_0: :
(dense18_matmul_readvariableop_resource_0: 7
)dense18_biasadd_readvariableop_resource_0:@
-combined_input_matmul_readvariableop_resource:	?	<
.combined_input_biasadd_readvariableop_resource:	@
.ensemble_output_matmul_readvariableop_resource:	=
/ensemble_output_biasadd_readvariableop_resource:
identity?? 1_dense18/BiasAdd/ReadVariableOp?1_dense18/MatMul/ReadVariableOp? 2_dense32/BiasAdd/ReadVariableOp?2_dense32/MatMul/ReadVariableOp? 3_dense64/BiasAdd/ReadVariableOp?3_dense64/MatMul/ReadVariableOp? 4_dense32/BiasAdd/ReadVariableOp?4_dense32/MatMul/ReadVariableOp? 5_dense18/BiasAdd/ReadVariableOp?5_dense18/MatMul/ReadVariableOp?$batch_normalization_5/AssignNewValue?&batch_normalization_5/AssignNewValue_1?5batch_normalization_5/FusedBatchNormV3/ReadVariableOp?7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_5/ReadVariableOp?&batch_normalization_5/ReadVariableOp_1?$batch_normalization_6/AssignNewValue?&batch_normalization_6/AssignNewValue_1?5batch_normalization_6/FusedBatchNormV3/ReadVariableOp?7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_6/ReadVariableOp?&batch_normalization_6/ReadVariableOp_1?$batch_normalization_7/AssignNewValue?&batch_normalization_7/AssignNewValue_1?5batch_normalization_7/FusedBatchNormV3/ReadVariableOp?7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_7/ReadVariableOp?&batch_normalization_7/ReadVariableOp_1?$batch_normalization_8/AssignNewValue?&batch_normalization_8/AssignNewValue_1?5batch_normalization_8/FusedBatchNormV3/ReadVariableOp?7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_8/ReadVariableOp?&batch_normalization_8/ReadVariableOp_1?$batch_normalization_9/AssignNewValue?&batch_normalization_9/AssignNewValue_1?5batch_normalization_9/FusedBatchNormV3/ReadVariableOp?7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_9/ReadVariableOp?&batch_normalization_9/ReadVariableOp_1?%combined_input/BiasAdd/ReadVariableOp?$combined_input/MatMul/ReadVariableOp?conv3d_5/BiasAdd/ReadVariableOp?conv3d_5/Conv3D/ReadVariableOp?conv3d_6/BiasAdd/ReadVariableOp?conv3d_6/Conv3D/ReadVariableOp?conv3d_7/BiasAdd/ReadVariableOp?conv3d_7/Conv3D/ReadVariableOp?conv3d_8/BiasAdd/ReadVariableOp?conv3d_8/Conv3D/ReadVariableOp?conv3d_9/BiasAdd/ReadVariableOp?conv3d_9/Conv3D/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?&ensemble_output/BiasAdd/ReadVariableOp?%ensemble_output/MatMul/ReadVariableOp?
conv3d_5/Conv3D/ReadVariableOpReadVariableOp'conv3d_5_conv3d_readvariableop_resource**
_output_shapes
: *
dtype02 
conv3d_5/Conv3D/ReadVariableOp?
conv3d_5/Conv3DConv3Dinputs_1&conv3d_5/Conv3D/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:??????????? *
paddingSAME*
strides	
2
conv3d_5/Conv3D?
conv3d_5/BiasAdd/ReadVariableOpReadVariableOp(conv3d_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv3d_5/BiasAdd/ReadVariableOp?
conv3d_5/BiasAddBiasAddconv3d_5/Conv3D:output:0'conv3d_5/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:??????????? 2
conv3d_5/BiasAdd?
conv3d_5/LeakyRelu	LeakyReluconv3d_5/BiasAdd:output:0*5
_output_shapes#
!:??????????? 2
conv3d_5/LeakyRelu?
max_pooling3d_5/MaxPool3D	MaxPool3D conv3d_5/LeakyRelu:activations:0*
T0*3
_output_shapes!
:?????????@@ *
ksize	
*
paddingVALID*
strides	
2
max_pooling3d_5/MaxPool3D?
$batch_normalization_5/ReadVariableOpReadVariableOp-batch_normalization_5_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_5/ReadVariableOp?
&batch_normalization_5/ReadVariableOp_1ReadVariableOp/batch_normalization_5_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_5/ReadVariableOp_1?
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3"max_pooling3d_5/MaxPool3D:output:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*O
_output_shapes=
;:?????????@@ : : : : :*
data_formatNDHWC*
epsilon%o?:*
exponential_avg_factor%
?#<2(
&batch_normalization_5/FusedBatchNormV3?
$batch_normalization_5/AssignNewValueAssignVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource3batch_normalization_5/FusedBatchNormV3:batch_mean:06^batch_normalization_5/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_5/AssignNewValue?
&batch_normalization_5/AssignNewValue_1AssignVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_5/FusedBatchNormV3:batch_variance:08^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_5/AssignNewValue_1?
conv3d_6/Conv3D/ReadVariableOpReadVariableOp'conv3d_6_conv3d_readvariableop_resource**
_output_shapes
:  *
dtype02 
conv3d_6/Conv3D/ReadVariableOp?
conv3d_6/Conv3DConv3D*batch_normalization_5/FusedBatchNormV3:y:0&conv3d_6/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@@ *
paddingSAME*
strides	
2
conv3d_6/Conv3D?
conv3d_6/BiasAdd/ReadVariableOpReadVariableOp(conv3d_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv3d_6/BiasAdd/ReadVariableOp?
conv3d_6/BiasAddBiasAddconv3d_6/Conv3D:output:0'conv3d_6/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@@ 2
conv3d_6/BiasAdd?
conv3d_6/LeakyRelu	LeakyReluconv3d_6/BiasAdd:output:0*3
_output_shapes!
:?????????@@ 2
conv3d_6/LeakyRelu?
max_pooling3d_6/MaxPool3D	MaxPool3D conv3d_6/LeakyRelu:activations:0*
T0*3
_output_shapes!
:?????????@@ *
ksize	
*
paddingVALID*
strides	
2
max_pooling3d_6/MaxPool3D?
$batch_normalization_6/ReadVariableOpReadVariableOp-batch_normalization_6_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_6/ReadVariableOp?
&batch_normalization_6/ReadVariableOp_1ReadVariableOp/batch_normalization_6_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_6/ReadVariableOp_1?
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_6/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_6/FusedBatchNormV3FusedBatchNormV3"max_pooling3d_6/MaxPool3D:output:0,batch_normalization_6/ReadVariableOp:value:0.batch_normalization_6/ReadVariableOp_1:value:0=batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*O
_output_shapes=
;:?????????@@ : : : : :*
data_formatNDHWC*
epsilon%o?:*
exponential_avg_factor%
?#<2(
&batch_normalization_6/FusedBatchNormV3?
$batch_normalization_6/AssignNewValueAssignVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource3batch_normalization_6/FusedBatchNormV3:batch_mean:06^batch_normalization_6/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_6/AssignNewValue?
&batch_normalization_6/AssignNewValue_1AssignVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_6/FusedBatchNormV3:batch_variance:08^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_6/AssignNewValue_1?
conv3d_7/Conv3D/ReadVariableOpReadVariableOp'conv3d_7_conv3d_readvariableop_resource**
_output_shapes
: @*
dtype02 
conv3d_7/Conv3D/ReadVariableOp?
conv3d_7/Conv3DConv3D*batch_normalization_6/FusedBatchNormV3:y:0&conv3d_7/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@@@*
paddingSAME*
strides	
2
conv3d_7/Conv3D?
conv3d_7/BiasAdd/ReadVariableOpReadVariableOp(conv3d_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv3d_7/BiasAdd/ReadVariableOp?
conv3d_7/BiasAddBiasAddconv3d_7/Conv3D:output:0'conv3d_7/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@@@2
conv3d_7/BiasAdd?
conv3d_7/LeakyRelu	LeakyReluconv3d_7/BiasAdd:output:0*3
_output_shapes!
:?????????@@@2
conv3d_7/LeakyRelu?
max_pooling3d_7/MaxPool3D	MaxPool3D conv3d_7/LeakyRelu:activations:0*
T0*3
_output_shapes!
:?????????@@@*
ksize	
*
paddingVALID*
strides	
2
max_pooling3d_7/MaxPool3D?
$batch_normalization_7/ReadVariableOpReadVariableOp-batch_normalization_7_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_7/ReadVariableOp?
&batch_normalization_7/ReadVariableOp_1ReadVariableOp/batch_normalization_7_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_7/ReadVariableOp_1?
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_7/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3"max_pooling3d_7/MaxPool3D:output:0,batch_normalization_7/ReadVariableOp:value:0.batch_normalization_7/ReadVariableOp_1:value:0=batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*O
_output_shapes=
;:?????????@@@:@:@:@:@:*
data_formatNDHWC*
epsilon%o?:*
exponential_avg_factor%
?#<2(
&batch_normalization_7/FusedBatchNormV3?
$batch_normalization_7/AssignNewValueAssignVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource3batch_normalization_7/FusedBatchNormV3:batch_mean:06^batch_normalization_7/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_7/AssignNewValue?
&batch_normalization_7/AssignNewValue_1AssignVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_7/FusedBatchNormV3:batch_variance:08^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_7/AssignNewValue_1?
conv3d_8/Conv3D/ReadVariableOpReadVariableOp'conv3d_8_conv3d_readvariableop_resource*+
_output_shapes
:@?*
dtype02 
conv3d_8/Conv3D/ReadVariableOp?
conv3d_8/Conv3DConv3D*batch_normalization_7/FusedBatchNormV3:y:0&conv3d_8/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :?????????@@?*
paddingSAME*
strides	
2
conv3d_8/Conv3D?
conv3d_8/BiasAdd/ReadVariableOpReadVariableOp(conv3d_8_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv3d_8/BiasAdd/ReadVariableOp?
conv3d_8/BiasAddBiasAddconv3d_8/Conv3D:output:0'conv3d_8/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :?????????@@?2
conv3d_8/BiasAdd?
conv3d_8/LeakyRelu	LeakyReluconv3d_8/BiasAdd:output:0*4
_output_shapes"
 :?????????@@?2
conv3d_8/LeakyRelu?
max_pooling3d_8/MaxPool3D	MaxPool3D conv3d_8/LeakyRelu:activations:0*
T0*4
_output_shapes"
 :?????????@@?*
ksize	
*
paddingVALID*
strides	
2
max_pooling3d_8/MaxPool3D?
$batch_normalization_8/ReadVariableOpReadVariableOp-batch_normalization_8_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$batch_normalization_8/ReadVariableOp?
&batch_normalization_8/ReadVariableOp_1ReadVariableOp/batch_normalization_8_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&batch_normalization_8/ReadVariableOp_1?
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype027
5batch_normalization_8/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype029
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_8/FusedBatchNormV3FusedBatchNormV3"max_pooling3d_8/MaxPool3D:output:0,batch_normalization_8/ReadVariableOp:value:0.batch_normalization_8/ReadVariableOp_1:value:0=batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*T
_output_shapesB
@:?????????@@?:?:?:?:?:*
data_formatNDHWC*
epsilon%o?:*
exponential_avg_factor%
?#<2(
&batch_normalization_8/FusedBatchNormV3?
$batch_normalization_8/AssignNewValueAssignVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource3batch_normalization_8/FusedBatchNormV3:batch_mean:06^batch_normalization_8/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_8/AssignNewValue?
&batch_normalization_8/AssignNewValue_1AssignVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_8/FusedBatchNormV3:batch_variance:08^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_8/AssignNewValue_1?
conv3d_9/Conv3D/ReadVariableOpReadVariableOp'conv3d_9_conv3d_readvariableop_resource*,
_output_shapes
:??*
dtype02 
conv3d_9/Conv3D/ReadVariableOp?
conv3d_9/Conv3DConv3D*batch_normalization_8/FusedBatchNormV3:y:0&conv3d_9/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :?????????@@?*
paddingSAME*
strides	
2
conv3d_9/Conv3D?
conv3d_9/BiasAdd/ReadVariableOpReadVariableOp(conv3d_9_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv3d_9/BiasAdd/ReadVariableOp?
conv3d_9/BiasAddBiasAddconv3d_9/Conv3D:output:0'conv3d_9/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :?????????@@?2
conv3d_9/BiasAdd?
conv3d_9/LeakyRelu	LeakyReluconv3d_9/BiasAdd:output:0*4
_output_shapes"
 :?????????@@?2
conv3d_9/LeakyRelu?
max_pooling3d_9/MaxPool3D	MaxPool3D conv3d_9/LeakyRelu:activations:0*
T0*4
_output_shapes"
 :?????????@@?*
ksize	
*
paddingVALID*
strides	
2
max_pooling3d_9/MaxPool3D?
1_dense18/MatMul/ReadVariableOpReadVariableOp&dense18_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02!
1_dense18/MatMul/ReadVariableOp?
1_dense18/MatMulMatMulinputs_0'1_dense18/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
1_dense18/MatMul?
 1_dense18/BiasAdd/ReadVariableOpReadVariableOp'dense18_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 1_dense18/BiasAdd/ReadVariableOp?
1_dense18/BiasAddBiasAdd1_dense18/MatMul:product:0(1_dense18/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
1_dense18/BiasAdd|
1_dense18/LeakyRelu	LeakyRelu1_dense18/BiasAdd:output:0*'
_output_shapes
:?????????2
1_dense18/LeakyRelu?
$batch_normalization_9/ReadVariableOpReadVariableOp-batch_normalization_9_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$batch_normalization_9/ReadVariableOp?
&batch_normalization_9/ReadVariableOp_1ReadVariableOp/batch_normalization_9_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&batch_normalization_9/ReadVariableOp_1?
5batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype027
5batch_normalization_9/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype029
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_9/FusedBatchNormV3FusedBatchNormV3"max_pooling3d_9/MaxPool3D:output:0,batch_normalization_9/ReadVariableOp:value:0.batch_normalization_9/ReadVariableOp_1:value:0=batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*T
_output_shapesB
@:?????????@@?:?:?:?:?:*
data_formatNDHWC*
epsilon%o?:*
exponential_avg_factor%
?#<2(
&batch_normalization_9/FusedBatchNormV3?
$batch_normalization_9/AssignNewValueAssignVariableOp>batch_normalization_9_fusedbatchnormv3_readvariableop_resource3batch_normalization_9/FusedBatchNormV3:batch_mean:06^batch_normalization_9/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_9/AssignNewValue?
&batch_normalization_9/AssignNewValue_1AssignVariableOp@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_9/FusedBatchNormV3:batch_variance:08^batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_9/AssignNewValue_1?
2_dense32/MatMul/ReadVariableOpReadVariableOp&dense32_matmul_readvariableop_resource*
_output_shapes

: *
dtype02!
2_dense32/MatMul/ReadVariableOp?
2_dense32/MatMulMatMul!1_dense18/LeakyRelu:activations:0'2_dense32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
2_dense32/MatMul?
 2_dense32/BiasAdd/ReadVariableOpReadVariableOp'dense32_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 2_dense32/BiasAdd/ReadVariableOp?
2_dense32/BiasAddBiasAdd2_dense32/MatMul:product:0(2_dense32/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
2_dense32/BiasAdd|
2_dense32/LeakyRelu	LeakyRelu2_dense32/BiasAdd:output:0*'
_output_shapes
:????????? 2
2_dense32/LeakyRelu?
1global_average_pooling3d_1/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         23
1global_average_pooling3d_1/Mean/reduction_indices?
global_average_pooling3d_1/MeanMean*batch_normalization_9/FusedBatchNormV3:y:0:global_average_pooling3d_1/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2!
global_average_pooling3d_1/Mean?
3_dense64/MatMul/ReadVariableOpReadVariableOp&dense64_matmul_readvariableop_resource*
_output_shapes

: @*
dtype02!
3_dense64/MatMul/ReadVariableOp?
3_dense64/MatMulMatMul!2_dense32/LeakyRelu:activations:0'3_dense64/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
3_dense64/MatMul?
 3_dense64/BiasAdd/ReadVariableOpReadVariableOp'dense64_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 3_dense64/BiasAdd/ReadVariableOp?
3_dense64/BiasAddBiasAdd3_dense64/MatMul:product:0(3_dense64/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
3_dense64/BiasAdd|
3_dense64/LeakyRelu	LeakyRelu3_dense64/BiasAdd:output:0*'
_output_shapes
:?????????@2
3_dense64/LeakyRelu?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMul(global_average_pooling3d_1/Mean:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_1/BiasAddw
dense_1/LeakyRelu	LeakyReludense_1/BiasAdd:output:0*(
_output_shapes
:??????????2
dense_1/LeakyRelu?
4_dense32/MatMul/ReadVariableOpReadVariableOp(dense32_matmul_readvariableop_resource_0*
_output_shapes

:@ *
dtype02!
4_dense32/MatMul/ReadVariableOp?
4_dense32/MatMulMatMul!3_dense64/LeakyRelu:activations:0'4_dense32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
4_dense32/MatMul?
 4_dense32/BiasAdd/ReadVariableOpReadVariableOp)dense32_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype02"
 4_dense32/BiasAdd/ReadVariableOp?
4_dense32/BiasAddBiasAdd4_dense32/MatMul:product:0(4_dense32/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
4_dense32/BiasAdd|
4_dense32/LeakyRelu	LeakyRelu4_dense32/BiasAdd:output:0*'
_output_shapes
:????????? 2
4_dense32/LeakyRelu?
5_dense18/MatMul/ReadVariableOpReadVariableOp(dense18_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype02!
5_dense18/MatMul/ReadVariableOp?
5_dense18/MatMulMatMul!4_dense32/LeakyRelu:activations:0'5_dense18/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
5_dense18/MatMul?
 5_dense18/BiasAdd/ReadVariableOpReadVariableOp)dense18_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype02"
 5_dense18/BiasAdd/ReadVariableOp?
5_dense18/BiasAddBiasAdd5_dense18/MatMul:product:0(5_dense18/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
5_dense18/BiasAdd|
5_dense18/LeakyRelu	LeakyRelu5_dense18/BiasAdd:output:0*'
_output_shapes
:?????????2
5_dense18/LeakyReluw
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout_1/dropout/Const?
dropout_1/dropout/MulMuldense_1/LeakyRelu:activations:0 dropout_1/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout_1/dropout/Mul?
dropout_1/dropout/ShapeShapedense_1/LeakyRelu:activations:0*
T0*
_output_shapes
:2
dropout_1/dropout/Shape?
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype020
.dropout_1/dropout/random_uniform/RandomUniform{
dropout_1/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *???>2
dropout_1/dropout/Const_1?
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0"dropout_1/dropout/Const_1:output:0*
T0*(
_output_shapes
:??????????2 
dropout_1/dropout/GreaterEqual?
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout_1/dropout/Cast?
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout_1/dropout/Mul_1x
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_1/concat/axis?
concatenate_1/concatConcatV2!5_dense18/LeakyRelu:activations:0dropout_1/dropout/Mul_1:z:0"concatenate_1/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
concatenate_1/concat?
$combined_input/MatMul/ReadVariableOpReadVariableOp-combined_input_matmul_readvariableop_resource*
_output_shapes
:	?	*
dtype02&
$combined_input/MatMul/ReadVariableOp?
combined_input/MatMulMatMulconcatenate_1/concat:output:0,combined_input/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2
combined_input/MatMul?
%combined_input/BiasAdd/ReadVariableOpReadVariableOp.combined_input_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02'
%combined_input/BiasAdd/ReadVariableOp?
combined_input/BiasAddBiasAddcombined_input/MatMul:product:0-combined_input/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2
combined_input/BiasAdd?
combined_input/LeakyRelu	LeakyRelucombined_input/BiasAdd:output:0*'
_output_shapes
:?????????	2
combined_input/LeakyRelu?
%ensemble_output/MatMul/ReadVariableOpReadVariableOp.ensemble_output_matmul_readvariableop_resource*
_output_shapes

:	*
dtype02'
%ensemble_output/MatMul/ReadVariableOp?
ensemble_output/MatMulMatMul&combined_input/LeakyRelu:activations:0-ensemble_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
ensemble_output/MatMul?
&ensemble_output/BiasAdd/ReadVariableOpReadVariableOp/ensemble_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&ensemble_output/BiasAdd/ReadVariableOp?
ensemble_output/BiasAddBiasAdd ensemble_output/MatMul:product:0.ensemble_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
ensemble_output/BiasAdd?
ensemble_output/SigmoidSigmoid ensemble_output/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
ensemble_output/Sigmoidv
IdentityIdentityensemble_output/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp!^1_dense18/BiasAdd/ReadVariableOp ^1_dense18/MatMul/ReadVariableOp!^2_dense32/BiasAdd/ReadVariableOp ^2_dense32/MatMul/ReadVariableOp!^3_dense64/BiasAdd/ReadVariableOp ^3_dense64/MatMul/ReadVariableOp!^4_dense32/BiasAdd/ReadVariableOp ^4_dense32/MatMul/ReadVariableOp!^5_dense18/BiasAdd/ReadVariableOp ^5_dense18/MatMul/ReadVariableOp%^batch_normalization_5/AssignNewValue'^batch_normalization_5/AssignNewValue_16^batch_normalization_5/FusedBatchNormV3/ReadVariableOp8^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_5/ReadVariableOp'^batch_normalization_5/ReadVariableOp_1%^batch_normalization_6/AssignNewValue'^batch_normalization_6/AssignNewValue_16^batch_normalization_6/FusedBatchNormV3/ReadVariableOp8^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_6/ReadVariableOp'^batch_normalization_6/ReadVariableOp_1%^batch_normalization_7/AssignNewValue'^batch_normalization_7/AssignNewValue_16^batch_normalization_7/FusedBatchNormV3/ReadVariableOp8^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_7/ReadVariableOp'^batch_normalization_7/ReadVariableOp_1%^batch_normalization_8/AssignNewValue'^batch_normalization_8/AssignNewValue_16^batch_normalization_8/FusedBatchNormV3/ReadVariableOp8^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_8/ReadVariableOp'^batch_normalization_8/ReadVariableOp_1%^batch_normalization_9/AssignNewValue'^batch_normalization_9/AssignNewValue_16^batch_normalization_9/FusedBatchNormV3/ReadVariableOp8^batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_9/ReadVariableOp'^batch_normalization_9/ReadVariableOp_1&^combined_input/BiasAdd/ReadVariableOp%^combined_input/MatMul/ReadVariableOp ^conv3d_5/BiasAdd/ReadVariableOp^conv3d_5/Conv3D/ReadVariableOp ^conv3d_6/BiasAdd/ReadVariableOp^conv3d_6/Conv3D/ReadVariableOp ^conv3d_7/BiasAdd/ReadVariableOp^conv3d_7/Conv3D/ReadVariableOp ^conv3d_8/BiasAdd/ReadVariableOp^conv3d_8/Conv3D/ReadVariableOp ^conv3d_9/BiasAdd/ReadVariableOp^conv3d_9/Conv3D/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp'^ensemble_output/BiasAdd/ReadVariableOp&^ensemble_output/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????
:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2D
 1_dense18/BiasAdd/ReadVariableOp 1_dense18/BiasAdd/ReadVariableOp2B
1_dense18/MatMul/ReadVariableOp1_dense18/MatMul/ReadVariableOp2D
 2_dense32/BiasAdd/ReadVariableOp 2_dense32/BiasAdd/ReadVariableOp2B
2_dense32/MatMul/ReadVariableOp2_dense32/MatMul/ReadVariableOp2D
 3_dense64/BiasAdd/ReadVariableOp 3_dense64/BiasAdd/ReadVariableOp2B
3_dense64/MatMul/ReadVariableOp3_dense64/MatMul/ReadVariableOp2D
 4_dense32/BiasAdd/ReadVariableOp 4_dense32/BiasAdd/ReadVariableOp2B
4_dense32/MatMul/ReadVariableOp4_dense32/MatMul/ReadVariableOp2D
 5_dense18/BiasAdd/ReadVariableOp 5_dense18/BiasAdd/ReadVariableOp2B
5_dense18/MatMul/ReadVariableOp5_dense18/MatMul/ReadVariableOp2L
$batch_normalization_5/AssignNewValue$batch_normalization_5/AssignNewValue2P
&batch_normalization_5/AssignNewValue_1&batch_normalization_5/AssignNewValue_12n
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp5batch_normalization_5/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_17batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_5/ReadVariableOp$batch_normalization_5/ReadVariableOp2P
&batch_normalization_5/ReadVariableOp_1&batch_normalization_5/ReadVariableOp_12L
$batch_normalization_6/AssignNewValue$batch_normalization_6/AssignNewValue2P
&batch_normalization_6/AssignNewValue_1&batch_normalization_6/AssignNewValue_12n
5batch_normalization_6/FusedBatchNormV3/ReadVariableOp5batch_normalization_6/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_17batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_6/ReadVariableOp$batch_normalization_6/ReadVariableOp2P
&batch_normalization_6/ReadVariableOp_1&batch_normalization_6/ReadVariableOp_12L
$batch_normalization_7/AssignNewValue$batch_normalization_7/AssignNewValue2P
&batch_normalization_7/AssignNewValue_1&batch_normalization_7/AssignNewValue_12n
5batch_normalization_7/FusedBatchNormV3/ReadVariableOp5batch_normalization_7/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_17batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_7/ReadVariableOp$batch_normalization_7/ReadVariableOp2P
&batch_normalization_7/ReadVariableOp_1&batch_normalization_7/ReadVariableOp_12L
$batch_normalization_8/AssignNewValue$batch_normalization_8/AssignNewValue2P
&batch_normalization_8/AssignNewValue_1&batch_normalization_8/AssignNewValue_12n
5batch_normalization_8/FusedBatchNormV3/ReadVariableOp5batch_normalization_8/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_17batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_8/ReadVariableOp$batch_normalization_8/ReadVariableOp2P
&batch_normalization_8/ReadVariableOp_1&batch_normalization_8/ReadVariableOp_12L
$batch_normalization_9/AssignNewValue$batch_normalization_9/AssignNewValue2P
&batch_normalization_9/AssignNewValue_1&batch_normalization_9/AssignNewValue_12n
5batch_normalization_9/FusedBatchNormV3/ReadVariableOp5batch_normalization_9/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_17batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_9/ReadVariableOp$batch_normalization_9/ReadVariableOp2P
&batch_normalization_9/ReadVariableOp_1&batch_normalization_9/ReadVariableOp_12N
%combined_input/BiasAdd/ReadVariableOp%combined_input/BiasAdd/ReadVariableOp2L
$combined_input/MatMul/ReadVariableOp$combined_input/MatMul/ReadVariableOp2B
conv3d_5/BiasAdd/ReadVariableOpconv3d_5/BiasAdd/ReadVariableOp2@
conv3d_5/Conv3D/ReadVariableOpconv3d_5/Conv3D/ReadVariableOp2B
conv3d_6/BiasAdd/ReadVariableOpconv3d_6/BiasAdd/ReadVariableOp2@
conv3d_6/Conv3D/ReadVariableOpconv3d_6/Conv3D/ReadVariableOp2B
conv3d_7/BiasAdd/ReadVariableOpconv3d_7/BiasAdd/ReadVariableOp2@
conv3d_7/Conv3D/ReadVariableOpconv3d_7/Conv3D/ReadVariableOp2B
conv3d_8/BiasAdd/ReadVariableOpconv3d_8/BiasAdd/ReadVariableOp2@
conv3d_8/Conv3D/ReadVariableOpconv3d_8/Conv3D/ReadVariableOp2B
conv3d_9/BiasAdd/ReadVariableOpconv3d_9/BiasAdd/ReadVariableOp2@
conv3d_9/Conv3D/ReadVariableOpconv3d_9/Conv3D/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2P
&ensemble_output/BiasAdd/ReadVariableOp&ensemble_output/BiasAdd/ReadVariableOp2N
%ensemble_output/MatMul/ReadVariableOp%ensemble_output/MatMul/ReadVariableOp:Q M
'
_output_shapes
:?????????

"
_user_specified_name
inputs/0:_[
5
_output_shapes#
!:???????????
"
_user_specified_name
inputs/1
??
?
G__inference_ensemble4d_layer_call_and_return_conditional_losses_9256922

svm_inputs
img3d_inputs.
conv3d_5_9256803: 
conv3d_5_9256805: +
batch_normalization_5_9256809: +
batch_normalization_5_9256811: +
batch_normalization_5_9256813: +
batch_normalization_5_9256815: .
conv3d_6_9256818:  
conv3d_6_9256820: +
batch_normalization_6_9256824: +
batch_normalization_6_9256826: +
batch_normalization_6_9256828: +
batch_normalization_6_9256830: .
conv3d_7_9256833: @
conv3d_7_9256835:@+
batch_normalization_7_9256839:@+
batch_normalization_7_9256841:@+
batch_normalization_7_9256843:@+
batch_normalization_7_9256845:@/
conv3d_8_9256848:@?
conv3d_8_9256850:	?,
batch_normalization_8_9256854:	?,
batch_normalization_8_9256856:	?,
batch_normalization_8_9256858:	?,
batch_normalization_8_9256860:	?0
conv3d_9_9256863:??
conv3d_9_9256865:	?!
dense18_9256869:

dense18_9256871:,
batch_normalization_9_9256874:	?,
batch_normalization_9_9256876:	?,
batch_normalization_9_9256878:	?,
batch_normalization_9_9256880:	?!
dense32_9256883: 
dense32_9256885: !
dense64_9256889: @
dense64_9256891:@#
dense_1_9256894:
??
dense_1_9256896:	?!
dense32_9256899:@ 
dense32_9256901: !
dense18_9256904: 
dense18_9256906:)
combined_input_9256911:	?	$
combined_input_9256913:	)
ensemble_output_9256916:	%
ensemble_output_9256918:
identity??!1_dense18/StatefulPartitionedCall?!2_dense32/StatefulPartitionedCall?!3_dense64/StatefulPartitionedCall?!4_dense32/StatefulPartitionedCall?!5_dense18/StatefulPartitionedCall?-batch_normalization_5/StatefulPartitionedCall?-batch_normalization_6/StatefulPartitionedCall?-batch_normalization_7/StatefulPartitionedCall?-batch_normalization_8/StatefulPartitionedCall?-batch_normalization_9/StatefulPartitionedCall?&combined_input/StatefulPartitionedCall? conv3d_5/StatefulPartitionedCall? conv3d_6/StatefulPartitionedCall? conv3d_7/StatefulPartitionedCall? conv3d_8/StatefulPartitionedCall? conv3d_9/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?'ensemble_output/StatefulPartitionedCall?
 conv3d_5/StatefulPartitionedCallStatefulPartitionedCallimg3d_inputsconv3d_5_9256803conv3d_5_9256805*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:??????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv3d_5_layer_call_and_return_conditional_losses_92553552"
 conv3d_5/StatefulPartitionedCall?
max_pooling3d_5/PartitionedCallPartitionedCall)conv3d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@@ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling3d_5_layer_call_and_return_conditional_losses_92553652!
max_pooling3d_5/PartitionedCall?
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_5/PartitionedCall:output:0batch_normalization_5_9256809batch_normalization_5_9256811batch_normalization_5_9256813batch_normalization_5_9256815*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@@ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_92562332/
-batch_normalization_5/StatefulPartitionedCall?
 conv3d_6/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0conv3d_6_9256818conv3d_6_9256820*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@@ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv3d_6_layer_call_and_return_conditional_losses_92554052"
 conv3d_6/StatefulPartitionedCall?
max_pooling3d_6/PartitionedCallPartitionedCall)conv3d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@@ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling3d_6_layer_call_and_return_conditional_losses_92554152!
max_pooling3d_6/PartitionedCall?
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_6/PartitionedCall:output:0batch_normalization_6_9256824batch_normalization_6_9256826batch_normalization_6_9256828batch_normalization_6_9256830*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@@ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_92561742/
-batch_normalization_6/StatefulPartitionedCall?
 conv3d_7/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0conv3d_7_9256833conv3d_7_9256835*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv3d_7_layer_call_and_return_conditional_losses_92554552"
 conv3d_7/StatefulPartitionedCall?
max_pooling3d_7/PartitionedCallPartitionedCall)conv3d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling3d_7_layer_call_and_return_conditional_losses_92554652!
max_pooling3d_7/PartitionedCall?
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_7/PartitionedCall:output:0batch_normalization_7_9256839batch_normalization_7_9256841batch_normalization_7_9256843batch_normalization_7_9256845*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_92561152/
-batch_normalization_7/StatefulPartitionedCall?
 conv3d_8/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0conv3d_8_9256848conv3d_8_9256850*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????@@?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv3d_8_layer_call_and_return_conditional_losses_92555052"
 conv3d_8/StatefulPartitionedCall?
max_pooling3d_8/PartitionedCallPartitionedCall)conv3d_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????@@?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling3d_8_layer_call_and_return_conditional_losses_92555152!
max_pooling3d_8/PartitionedCall?
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_8/PartitionedCall:output:0batch_normalization_8_9256854batch_normalization_8_9256856batch_normalization_8_9256858batch_normalization_8_9256860*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????@@?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_92560562/
-batch_normalization_8/StatefulPartitionedCall?
 conv3d_9/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0conv3d_9_9256863conv3d_9_9256865*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????@@?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv3d_9_layer_call_and_return_conditional_losses_92555552"
 conv3d_9/StatefulPartitionedCall?
max_pooling3d_9/PartitionedCallPartitionedCall)conv3d_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????@@?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling3d_9_layer_call_and_return_conditional_losses_92555652!
max_pooling3d_9/PartitionedCall?
!1_dense18/StatefulPartitionedCallStatefulPartitionedCall
svm_inputsdense18_9256869dense18_9256871*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_1_dense18_layer_call_and_return_conditional_losses_92555782#
!1_dense18/StatefulPartitionedCall?
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_9/PartitionedCall:output:0batch_normalization_9_9256874batch_normalization_9_9256876batch_normalization_9_9256878batch_normalization_9_9256880*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????@@?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_92559872/
-batch_normalization_9/StatefulPartitionedCall?
!2_dense32/StatefulPartitionedCallStatefulPartitionedCall*1_dense18/StatefulPartitionedCall:output:0dense32_9256883dense32_9256885*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_2_dense32_layer_call_and_return_conditional_losses_92556222#
!2_dense32/StatefulPartitionedCall?
*global_average_pooling3d_1/PartitionedCallPartitionedCall6batch_normalization_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *`
f[RY
W__inference_global_average_pooling3d_1_layer_call_and_return_conditional_losses_92556332,
*global_average_pooling3d_1/PartitionedCall?
!3_dense64/StatefulPartitionedCallStatefulPartitionedCall*2_dense32/StatefulPartitionedCall:output:0dense64_9256889dense64_9256891*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_3_dense64_layer_call_and_return_conditional_losses_92556462#
!3_dense64/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling3d_1/PartitionedCall:output:0dense_1_9256894dense_1_9256896*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_92556632!
dense_1/StatefulPartitionedCall?
!4_dense32/StatefulPartitionedCallStatefulPartitionedCall*3_dense64/StatefulPartitionedCall:output:0dense32_9256899dense32_9256901*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_4_dense32_layer_call_and_return_conditional_losses_92556802#
!4_dense32/StatefulPartitionedCall?
!5_dense18/StatefulPartitionedCallStatefulPartitionedCall*4_dense32/StatefulPartitionedCall:output:0dense18_9256904dense18_9256906*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_5_dense18_layer_call_and_return_conditional_losses_92556972#
!5_dense18/StatefulPartitionedCall?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_92558962#
!dropout_1/StatefulPartitionedCall?
concatenate_1/PartitionedCallPartitionedCall*5_dense18/StatefulPartitionedCall:output:0*dropout_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_concatenate_1_layer_call_and_return_conditional_losses_92557172
concatenate_1/PartitionedCall?
&combined_input/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0combined_input_9256911combined_input_9256913*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_combined_input_layer_call_and_return_conditional_losses_92557302(
&combined_input/StatefulPartitionedCall?
'ensemble_output/StatefulPartitionedCallStatefulPartitionedCall/combined_input/StatefulPartitionedCall:output:0ensemble_output_9256916ensemble_output_9256918*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_ensemble_output_layer_call_and_return_conditional_losses_92557472)
'ensemble_output/StatefulPartitionedCall?
IdentityIdentity0ensemble_output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp"^1_dense18/StatefulPartitionedCall"^2_dense32/StatefulPartitionedCall"^3_dense64/StatefulPartitionedCall"^4_dense32/StatefulPartitionedCall"^5_dense18/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall.^batch_normalization_9/StatefulPartitionedCall'^combined_input/StatefulPartitionedCall!^conv3d_5/StatefulPartitionedCall!^conv3d_6/StatefulPartitionedCall!^conv3d_7/StatefulPartitionedCall!^conv3d_8/StatefulPartitionedCall!^conv3d_9/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall(^ensemble_output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????
:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2F
!1_dense18/StatefulPartitionedCall!1_dense18/StatefulPartitionedCall2F
!2_dense32/StatefulPartitionedCall!2_dense32/StatefulPartitionedCall2F
!3_dense64/StatefulPartitionedCall!3_dense64/StatefulPartitionedCall2F
!4_dense32/StatefulPartitionedCall!4_dense32/StatefulPartitionedCall2F
!5_dense18/StatefulPartitionedCall!5_dense18/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2P
&combined_input/StatefulPartitionedCall&combined_input/StatefulPartitionedCall2D
 conv3d_5/StatefulPartitionedCall conv3d_5/StatefulPartitionedCall2D
 conv3d_6/StatefulPartitionedCall conv3d_6/StatefulPartitionedCall2D
 conv3d_7/StatefulPartitionedCall conv3d_7/StatefulPartitionedCall2D
 conv3d_8/StatefulPartitionedCall conv3d_8/StatefulPartitionedCall2D
 conv3d_9/StatefulPartitionedCall conv3d_9/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2R
'ensemble_output/StatefulPartitionedCall'ensemble_output/StatefulPartitionedCall:S O
'
_output_shapes
:?????????

$
_user_specified_name
svm_inputs:c_
5
_output_shapes#
!:???????????
&
_user_specified_nameimg3d_inputs
?
?
,__inference_ensemble4d_layer_call_fn_9256676

svm_inputs
img3d_inputs%
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: '
	unknown_5:  
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: (

unknown_11: @

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@)

unknown_17:@?

unknown_18:	?

unknown_19:	?

unknown_20:	?

unknown_21:	?

unknown_22:	?*

unknown_23:??

unknown_24:	?

unknown_25:


unknown_26:

unknown_27:	?

unknown_28:	?

unknown_29:	?

unknown_30:	?

unknown_31: 

unknown_32: 

unknown_33: @

unknown_34:@

unknown_35:
??

unknown_36:	?

unknown_37:@ 

unknown_38: 

unknown_39: 

unknown_40:

unknown_41:	?	

unknown_42:	

unknown_43:	

unknown_44:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall
svm_inputsimg3d_inputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44*;
Tin4
220*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*F
_read_only_resource_inputs(
&$	
"#$%&'()*+,-./*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_ensemble4d_layer_call_and_return_conditional_losses_92564832
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????
:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
'
_output_shapes
:?????????

$
_user_specified_name
svm_inputs:c_
5
_output_shapes#
!:???????????
&
_user_specified_nameimg3d_inputs
?
h
L__inference_max_pooling3d_7_layer_call_and_return_conditional_losses_9254876

inputs
identity?
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????*
ksize	
*
paddingVALID*
strides	
2
	MaxPool3D?
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A?????????????????????????????????????????????: {
W
_output_shapesE
C:A?????????????????????????????????????????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_9254763

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*j
_output_shapesX
V:8???????????????????????????????????? : : : : :*
data_formatNDHWC*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*N
_output_shapes<
::8???????????????????????????????????? 2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8???????????????????????????????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:v r
N
_output_shapes<
::8???????????????????????????????????? 
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_9258055

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*O
_output_shapes=
;:?????????@@@:@:@:@:@:*
data_formatNDHWC*
epsilon%o?:*
is_training( 2
FusedBatchNormV3{
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*3
_output_shapes!
:?????????@@@2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????@@@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:[ W
3
_output_shapes!
:?????????@@@
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_9258073

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*O
_output_shapes=
;:?????????@@@:@:@:@:@:*
data_formatNDHWC*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1{
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*3
_output_shapes!
:?????????@@@2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????@@@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:[ W
3
_output_shapes!
:?????????@@@
 
_user_specified_nameinputs
?
?
+__inference_5_dense18_layer_call_fn_9258532

inputs
unknown: 
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_5_dense18_layer_call_and_return_conditional_losses_92556972
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?	
?
7__inference_batch_normalization_6_layer_call_fn_9257824

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@@ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_92554342
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:?????????@@ 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????@@ : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:?????????@@ 
 
_user_specified_nameinputs
?
?
L__inference_ensemble_output_layer_call_and_return_conditional_losses_9255747

inputs0
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoidf
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
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
?
?
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_9256174

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*O
_output_shapes=
;:?????????@@ : : : : :*
data_formatNDHWC*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1{
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*3
_output_shapes!
:?????????@@ 2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????@@ : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:[ W
3
_output_shapes!
:?????????@@ 
 
_user_specified_nameinputs
??
?
G__inference_ensemble4d_layer_call_and_return_conditional_losses_9256799

svm_inputs
img3d_inputs.
conv3d_5_9256680: 
conv3d_5_9256682: +
batch_normalization_5_9256686: +
batch_normalization_5_9256688: +
batch_normalization_5_9256690: +
batch_normalization_5_9256692: .
conv3d_6_9256695:  
conv3d_6_9256697: +
batch_normalization_6_9256701: +
batch_normalization_6_9256703: +
batch_normalization_6_9256705: +
batch_normalization_6_9256707: .
conv3d_7_9256710: @
conv3d_7_9256712:@+
batch_normalization_7_9256716:@+
batch_normalization_7_9256718:@+
batch_normalization_7_9256720:@+
batch_normalization_7_9256722:@/
conv3d_8_9256725:@?
conv3d_8_9256727:	?,
batch_normalization_8_9256731:	?,
batch_normalization_8_9256733:	?,
batch_normalization_8_9256735:	?,
batch_normalization_8_9256737:	?0
conv3d_9_9256740:??
conv3d_9_9256742:	?!
dense18_9256746:

dense18_9256748:,
batch_normalization_9_9256751:	?,
batch_normalization_9_9256753:	?,
batch_normalization_9_9256755:	?,
batch_normalization_9_9256757:	?!
dense32_9256760: 
dense32_9256762: !
dense64_9256766: @
dense64_9256768:@#
dense_1_9256771:
??
dense_1_9256773:	?!
dense32_9256776:@ 
dense32_9256778: !
dense18_9256781: 
dense18_9256783:)
combined_input_9256788:	?	$
combined_input_9256790:	)
ensemble_output_9256793:	%
ensemble_output_9256795:
identity??!1_dense18/StatefulPartitionedCall?!2_dense32/StatefulPartitionedCall?!3_dense64/StatefulPartitionedCall?!4_dense32/StatefulPartitionedCall?!5_dense18/StatefulPartitionedCall?-batch_normalization_5/StatefulPartitionedCall?-batch_normalization_6/StatefulPartitionedCall?-batch_normalization_7/StatefulPartitionedCall?-batch_normalization_8/StatefulPartitionedCall?-batch_normalization_9/StatefulPartitionedCall?&combined_input/StatefulPartitionedCall? conv3d_5/StatefulPartitionedCall? conv3d_6/StatefulPartitionedCall? conv3d_7/StatefulPartitionedCall? conv3d_8/StatefulPartitionedCall? conv3d_9/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?'ensemble_output/StatefulPartitionedCall?
 conv3d_5/StatefulPartitionedCallStatefulPartitionedCallimg3d_inputsconv3d_5_9256680conv3d_5_9256682*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:??????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv3d_5_layer_call_and_return_conditional_losses_92553552"
 conv3d_5/StatefulPartitionedCall?
max_pooling3d_5/PartitionedCallPartitionedCall)conv3d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@@ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling3d_5_layer_call_and_return_conditional_losses_92553652!
max_pooling3d_5/PartitionedCall?
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_5/PartitionedCall:output:0batch_normalization_5_9256686batch_normalization_5_9256688batch_normalization_5_9256690batch_normalization_5_9256692*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@@ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_92553842/
-batch_normalization_5/StatefulPartitionedCall?
 conv3d_6/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0conv3d_6_9256695conv3d_6_9256697*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@@ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv3d_6_layer_call_and_return_conditional_losses_92554052"
 conv3d_6/StatefulPartitionedCall?
max_pooling3d_6/PartitionedCallPartitionedCall)conv3d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@@ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling3d_6_layer_call_and_return_conditional_losses_92554152!
max_pooling3d_6/PartitionedCall?
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_6/PartitionedCall:output:0batch_normalization_6_9256701batch_normalization_6_9256703batch_normalization_6_9256705batch_normalization_6_9256707*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@@ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_92554342/
-batch_normalization_6/StatefulPartitionedCall?
 conv3d_7/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0conv3d_7_9256710conv3d_7_9256712*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv3d_7_layer_call_and_return_conditional_losses_92554552"
 conv3d_7/StatefulPartitionedCall?
max_pooling3d_7/PartitionedCallPartitionedCall)conv3d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling3d_7_layer_call_and_return_conditional_losses_92554652!
max_pooling3d_7/PartitionedCall?
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_7/PartitionedCall:output:0batch_normalization_7_9256716batch_normalization_7_9256718batch_normalization_7_9256720batch_normalization_7_9256722*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@@@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_92554842/
-batch_normalization_7/StatefulPartitionedCall?
 conv3d_8/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0conv3d_8_9256725conv3d_8_9256727*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????@@?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv3d_8_layer_call_and_return_conditional_losses_92555052"
 conv3d_8/StatefulPartitionedCall?
max_pooling3d_8/PartitionedCallPartitionedCall)conv3d_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????@@?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling3d_8_layer_call_and_return_conditional_losses_92555152!
max_pooling3d_8/PartitionedCall?
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_8/PartitionedCall:output:0batch_normalization_8_9256731batch_normalization_8_9256733batch_normalization_8_9256735batch_normalization_8_9256737*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????@@?*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_92555342/
-batch_normalization_8/StatefulPartitionedCall?
 conv3d_9/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0conv3d_9_9256740conv3d_9_9256742*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????@@?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv3d_9_layer_call_and_return_conditional_losses_92555552"
 conv3d_9/StatefulPartitionedCall?
max_pooling3d_9/PartitionedCallPartitionedCall)conv3d_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????@@?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling3d_9_layer_call_and_return_conditional_losses_92555652!
max_pooling3d_9/PartitionedCall?
!1_dense18/StatefulPartitionedCallStatefulPartitionedCall
svm_inputsdense18_9256746dense18_9256748*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_1_dense18_layer_call_and_return_conditional_losses_92555782#
!1_dense18/StatefulPartitionedCall?
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_9/PartitionedCall:output:0batch_normalization_9_9256751batch_normalization_9_9256753batch_normalization_9_9256755batch_normalization_9_9256757*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????@@?*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_92556012/
-batch_normalization_9/StatefulPartitionedCall?
!2_dense32/StatefulPartitionedCallStatefulPartitionedCall*1_dense18/StatefulPartitionedCall:output:0dense32_9256760dense32_9256762*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_2_dense32_layer_call_and_return_conditional_losses_92556222#
!2_dense32/StatefulPartitionedCall?
*global_average_pooling3d_1/PartitionedCallPartitionedCall6batch_normalization_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *`
f[RY
W__inference_global_average_pooling3d_1_layer_call_and_return_conditional_losses_92556332,
*global_average_pooling3d_1/PartitionedCall?
!3_dense64/StatefulPartitionedCallStatefulPartitionedCall*2_dense32/StatefulPartitionedCall:output:0dense64_9256766dense64_9256768*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_3_dense64_layer_call_and_return_conditional_losses_92556462#
!3_dense64/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling3d_1/PartitionedCall:output:0dense_1_9256771dense_1_9256773*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_92556632!
dense_1/StatefulPartitionedCall?
!4_dense32/StatefulPartitionedCallStatefulPartitionedCall*3_dense64/StatefulPartitionedCall:output:0dense32_9256776dense32_9256778*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_4_dense32_layer_call_and_return_conditional_losses_92556802#
!4_dense32/StatefulPartitionedCall?
!5_dense18/StatefulPartitionedCallStatefulPartitionedCall*4_dense32/StatefulPartitionedCall:output:0dense18_9256781dense18_9256783*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_5_dense18_layer_call_and_return_conditional_losses_92556972#
!5_dense18/StatefulPartitionedCall?
dropout_1/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_92557082
dropout_1/PartitionedCall?
concatenate_1/PartitionedCallPartitionedCall*5_dense18/StatefulPartitionedCall:output:0"dropout_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_concatenate_1_layer_call_and_return_conditional_losses_92557172
concatenate_1/PartitionedCall?
&combined_input/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0combined_input_9256788combined_input_9256790*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_combined_input_layer_call_and_return_conditional_losses_92557302(
&combined_input/StatefulPartitionedCall?
'ensemble_output/StatefulPartitionedCallStatefulPartitionedCall/combined_input/StatefulPartitionedCall:output:0ensemble_output_9256793ensemble_output_9256795*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_ensemble_output_layer_call_and_return_conditional_losses_92557472)
'ensemble_output/StatefulPartitionedCall?
IdentityIdentity0ensemble_output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp"^1_dense18/StatefulPartitionedCall"^2_dense32/StatefulPartitionedCall"^3_dense64/StatefulPartitionedCall"^4_dense32/StatefulPartitionedCall"^5_dense18/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall.^batch_normalization_9/StatefulPartitionedCall'^combined_input/StatefulPartitionedCall!^conv3d_5/StatefulPartitionedCall!^conv3d_6/StatefulPartitionedCall!^conv3d_7/StatefulPartitionedCall!^conv3d_8/StatefulPartitionedCall!^conv3d_9/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall(^ensemble_output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????
:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2F
!1_dense18/StatefulPartitionedCall!1_dense18/StatefulPartitionedCall2F
!2_dense32/StatefulPartitionedCall!2_dense32/StatefulPartitionedCall2F
!3_dense64/StatefulPartitionedCall!3_dense64/StatefulPartitionedCall2F
!4_dense32/StatefulPartitionedCall!4_dense32/StatefulPartitionedCall2F
!5_dense18/StatefulPartitionedCall!5_dense18/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2P
&combined_input/StatefulPartitionedCall&combined_input/StatefulPartitionedCall2D
 conv3d_5/StatefulPartitionedCall conv3d_5/StatefulPartitionedCall2D
 conv3d_6/StatefulPartitionedCall conv3d_6/StatefulPartitionedCall2D
 conv3d_7/StatefulPartitionedCall conv3d_7/StatefulPartitionedCall2D
 conv3d_8/StatefulPartitionedCall conv3d_8/StatefulPartitionedCall2D
 conv3d_9/StatefulPartitionedCall conv3d_9/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2R
'ensemble_output/StatefulPartitionedCall'ensemble_output/StatefulPartitionedCall:S O
'
_output_shapes
:?????????

$
_user_specified_name
svm_inputs:c_
5
_output_shapes#
!:???????????
&
_user_specified_nameimg3d_inputs
?
?
*__inference_conv3d_5_layer_call_fn_9257590

inputs%
unknown: 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:??????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv3d_5_layer_call_and_return_conditional_losses_92553552
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:??????????? 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:???????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:???????????
 
_user_specified_nameinputs
?
M
1__inference_max_pooling3d_7_layer_call_fn_9257934

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:A?????????????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling3d_7_layer_call_and_return_conditional_losses_92548762
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A?????????????????????????????????????????????: {
W
_output_shapesE
C:A?????????????????????????????????????????????
 
_user_specified_nameinputs
?
h
L__inference_max_pooling3d_8_layer_call_and_return_conditional_losses_9258113

inputs
identity?
	MaxPool3D	MaxPool3Dinputs*
T0*4
_output_shapes"
 :?????????@@?*
ksize	
*
paddingVALID*
strides	
2
	MaxPool3Ds
IdentityIdentityMaxPool3D:output:0*
T0*4
_output_shapes"
 :?????????@@?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????@@?:\ X
4
_output_shapes"
 :?????????@@?
 
_user_specified_nameinputs
?
d
F__inference_dropout_1_layer_call_and_return_conditional_losses_9255708

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_9256115

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*O
_output_shapes=
;:?????????@@@:@:@:@:@:*
data_formatNDHWC*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1{
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*3
_output_shapes!
:?????????@@@2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????@@@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:[ W
3
_output_shapes!
:?????????@@@
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_9255103

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*o
_output_shapes]
[:9?????????????????????????????????????:?:?:?:?:*
data_formatNDHWC*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*O
_output_shapes=
;:9?????????????????????????????????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:9?????????????????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:w s
O
_output_shapes=
;:9?????????????????????????????????????
 
_user_specified_nameinputs
?
h
L__inference_max_pooling3d_5_layer_call_and_return_conditional_losses_9257621

inputs
identity?
	MaxPool3D	MaxPool3Dinputs*
T0*3
_output_shapes!
:?????????@@ *
ksize	
*
paddingVALID*
strides	
2
	MaxPool3Dr
IdentityIdentityMaxPool3D:output:0*
T0*3
_output_shapes!
:?????????@@ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:??????????? :] Y
5
_output_shapes#
!:??????????? 
 
_user_specified_nameinputs
?
?
D__inference_dense_1_layer_call_and_return_conditional_losses_9258523

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd_
	LeakyRelu	LeakyReluBiasAdd:output:0*(
_output_shapes
:??????????2
	LeakyRelus
IdentityIdentityLeakyRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
F__inference_2_dense32_layer_call_and_return_conditional_losses_9258317

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAdd^
	LeakyRelu	LeakyReluBiasAdd:output:0*'
_output_shapes
:????????? 2
	LeakyRelur
IdentityIdentityLeakyRelu:activations:0^NoOp*
T0*'
_output_shapes
:????????? 2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_9254911

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*j
_output_shapesX
V:8????????????????????????????????????@:@:@:@:@:*
data_formatNDHWC*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*N
_output_shapes<
::8????????????????????????????????????@2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8????????????????????????????????????@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:v r
N
_output_shapes<
::8????????????????????????????????????@
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_9258183

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*o
_output_shapes]
[:9?????????????????????????????????????:?:?:?:?:*
data_formatNDHWC*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*O
_output_shapes=
;:9?????????????????????????????????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:9?????????????????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:w s
O
_output_shapes=
;:9?????????????????????????????????????
 
_user_specified_nameinputs
?
h
L__inference_max_pooling3d_6_layer_call_and_return_conditional_losses_9257785

inputs
identity?
	MaxPool3D	MaxPool3Dinputs*
T0*3
_output_shapes!
:?????????@@ *
ksize	
*
paddingVALID*
strides	
2
	MaxPool3Dr
IdentityIdentityMaxPool3D:output:0*
T0*3
_output_shapes!
:?????????@@ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@ :[ W
3
_output_shapes!
:?????????@@ 
 
_user_specified_nameinputs
?
h
L__inference_max_pooling3d_6_layer_call_and_return_conditional_losses_9255415

inputs
identity?
	MaxPool3D	MaxPool3Dinputs*
T0*3
_output_shapes!
:?????????@@ *
ksize	
*
paddingVALID*
strides	
2
	MaxPool3Dr
IdentityIdentityMaxPool3D:output:0*
T0*3
_output_shapes!
:?????????@@ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@ :[ W
3
_output_shapes!
:?????????@@ 
 
_user_specified_nameinputs
?
h
L__inference_max_pooling3d_7_layer_call_and_return_conditional_losses_9255465

inputs
identity?
	MaxPool3D	MaxPool3Dinputs*
T0*3
_output_shapes!
:?????????@@@*
ksize	
*
paddingVALID*
strides	
2
	MaxPool3Dr
IdentityIdentityMaxPool3D:output:0*
T0*3
_output_shapes!
:?????????@@@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@@:[ W
3
_output_shapes!
:?????????@@@
 
_user_specified_nameinputs
?	
?
7__inference_batch_normalization_8_layer_call_fn_9258152

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????@@?*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_92555342
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :?????????@@?2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:?????????@@?: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :?????????@@?
 
_user_specified_nameinputs
?
?
K__inference_combined_input_layer_call_and_return_conditional_losses_9258603

inputs1
matmul_readvariableop_resource:	?	-
biasadd_readvariableop_resource:	
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2	
BiasAdd^
	LeakyRelu	LeakyReluBiasAdd:output:0*'
_output_shapes
:?????????	2
	LeakyRelur
IdentityIdentityLeakyRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????	2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_9254955

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*j
_output_shapesX
V:8????????????????????????????????????@:@:@:@:@:*
data_formatNDHWC*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*N
_output_shapes<
::8????????????????????????????????????@2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8????????????????????????????????????@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:v r
N
_output_shapes<
::8????????????????????????????????????@
 
_user_specified_nameinputs
?
h
L__inference_max_pooling3d_6_layer_call_and_return_conditional_losses_9254728

inputs
identity?
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????*
ksize	
*
paddingVALID*
strides	
2
	MaxPool3D?
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A?????????????????????????????????????????????: {
W
_output_shapesE
C:A?????????????????????????????????????????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_9255384

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*O
_output_shapes=
;:?????????@@ : : : : :*
data_formatNDHWC*
epsilon%o?:*
is_training( 2
FusedBatchNormV3{
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*3
_output_shapes!
:?????????@@ 2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????@@ : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:[ W
3
_output_shapes!
:?????????@@ 
 
_user_specified_nameinputs
?
?
,__inference_ensemble4d_layer_call_fn_9257222
inputs_0
inputs_1%
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: '
	unknown_5:  
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: (

unknown_11: @

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@)

unknown_17:@?

unknown_18:	?

unknown_19:	?

unknown_20:	?

unknown_21:	?

unknown_22:	?*

unknown_23:??

unknown_24:	?

unknown_25:


unknown_26:

unknown_27:	?

unknown_28:	?

unknown_29:	?

unknown_30:	?

unknown_31: 

unknown_32: 

unknown_33: @

unknown_34:@

unknown_35:
??

unknown_36:	?

unknown_37:@ 

unknown_38: 

unknown_39: 

unknown_40:

unknown_41:	?	

unknown_42:	

unknown_43:	

unknown_44:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44*;
Tin4
220*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*F
_read_only_resource_inputs(
&$	
"#$%&'()*+,-./*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_ensemble4d_layer_call_and_return_conditional_losses_92564832
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????
:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????

"
_user_specified_name
inputs/0:_[
5
_output_shapes#
!:???????????
"
_user_specified_name
inputs/1
?
?
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_9257891

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*O
_output_shapes=
;:?????????@@ : : : : :*
data_formatNDHWC*
epsilon%o?:*
is_training( 2
FusedBatchNormV3{
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*3
_output_shapes!
:?????????@@ 2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????@@ : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:[ W
3
_output_shapes!
:?????????@@ 
 
_user_specified_nameinputs
?
?
F__inference_1_dense18_layer_call_and_return_conditional_losses_9258277

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd^
	LeakyRelu	LeakyReluBiasAdd:output:0*'
_output_shapes
:?????????2
	LeakyRelur
IdentityIdentityLeakyRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
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
7__inference_batch_normalization_9_layer_call_fn_9258356

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????@@?*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_92556012
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :?????????@@?2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:?????????@@?: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :?????????@@?
 
_user_specified_nameinputs
?	
?
7__inference_batch_normalization_5_layer_call_fn_9257673

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@@ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_92562332
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:?????????@@ 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????@@ : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:?????????@@ 
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_9257709

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*j
_output_shapesX
V:8???????????????????????????????????? : : : : :*
data_formatNDHWC*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*N
_output_shapes<
::8???????????????????????????????????? 2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8???????????????????????????????????? : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:v r
N
_output_shapes<
::8???????????????????????????????????? 
 
_user_specified_nameinputs
?
M
1__inference_max_pooling3d_7_layer_call_fn_9257939

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling3d_7_layer_call_and_return_conditional_losses_92554652
PartitionedCallx
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:?????????@@@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@@:[ W
3
_output_shapes!
:?????????@@@
 
_user_specified_nameinputs
?
M
1__inference_max_pooling3d_5_layer_call_fn_9257606

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:A?????????????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling3d_5_layer_call_and_return_conditional_losses_92545802
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A?????????????????????????????????????????????: {
W
_output_shapesE
C:A?????????????????????????????????????????????
 
_user_specified_nameinputs
?	
?
7__inference_batch_normalization_5_layer_call_fn_9257647

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8???????????????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_92546592
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*N
_output_shapes<
::8???????????????????????????????????? 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8???????????????????????????????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8???????????????????????????????????? 
 
_user_specified_nameinputs
?
M
1__inference_max_pooling3d_9_layer_call_fn_9258282

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:A?????????????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling3d_9_layer_call_and_return_conditional_losses_92551722
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A?????????????????????????????????????????????: {
W
_output_shapesE
C:A?????????????????????????????????????????????
 
_user_specified_nameinputs
?
?
0__inference_combined_input_layer_call_fn_9258592

inputs
unknown:	?	
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
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_combined_input_layer_call_and_return_conditional_losses_92557302
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????	2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
t
J__inference_concatenate_1_layer_call_and_return_conditional_losses_9255717

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????:??????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
,__inference_ensemble4d_layer_call_fn_9255849

svm_inputs
img3d_inputs%
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: '
	unknown_5:  
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: (

unknown_11: @

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@)

unknown_17:@?

unknown_18:	?

unknown_19:	?

unknown_20:	?

unknown_21:	?

unknown_22:	?*

unknown_23:??

unknown_24:	?

unknown_25:


unknown_26:

unknown_27:	?

unknown_28:	?

unknown_29:	?

unknown_30:	?

unknown_31: 

unknown_32: 

unknown_33: @

unknown_34:@

unknown_35:
??

unknown_36:	?

unknown_37:@ 

unknown_38: 

unknown_39: 

unknown_40:

unknown_41:	?	

unknown_42:	

unknown_43:	

unknown_44:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall
svm_inputsimg3d_inputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44*;
Tin4
220*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*P
_read_only_resource_inputs2
0.	
 !"#$%&'()*+,-./*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_ensemble4d_layer_call_and_return_conditional_losses_92557542
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????
:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
'
_output_shapes
:?????????

$
_user_specified_name
svm_inputs:c_
5
_output_shapes#
!:???????????
&
_user_specified_nameimg3d_inputs
?

?
7__inference_batch_normalization_8_layer_call_fn_9258126

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *O
_output_shapes=
;:9?????????????????????????????????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_92550592
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*O
_output_shapes=
;:9?????????????????????????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:9?????????????????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:w s
O
_output_shapes=
;:9?????????????????????????????????????
 
_user_specified_nameinputs
?
?
*__inference_conv3d_9_layer_call_fn_9258246

inputs'
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????@@?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv3d_9_layer_call_and_return_conditional_losses_92555552
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :?????????@@?2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????@@?: : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :?????????@@?
 
_user_specified_nameinputs
?	
?
7__inference_batch_normalization_6_layer_call_fn_9257811

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8???????????????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_92548072
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*N
_output_shapes<
::8???????????????????????????????????? 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8???????????????????????????????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8???????????????????????????????????? 
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
S
img3d_inputsC
serving_default_img3d_inputs:0???????????
A

svm_inputs3
serving_default_svm_inputs:0?????????
C
ensemble_output0
StatefulPartitionedCall:0?????????tensorflow/serving/predict:
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
	layer-8

layer_with_weights-5

layer-9
layer_with_weights-6
layer-10
layer-11
layer_with_weights-7
layer-12
layer-13
layer_with_weights-8
layer-14
layer_with_weights-9
layer-15
layer-16
layer_with_weights-10
layer-17
layer_with_weights-11
layer-18
layer_with_weights-12
layer-19
layer-20
layer_with_weights-13
layer-21
layer_with_weights-14
layer-22
layer_with_weights-15
layer-23
layer-24
layer-25
layer_with_weights-16
layer-26
layer_with_weights-17
layer-27
	optimizer
	variables
trainable_variables
 regularization_losses
!	keras_api
"
signatures
?_default_save_signature
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_network
"
_tf_keras_input_layer
?

#kernel
$bias
%	variables
&trainable_variables
'regularization_losses
(	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
)	variables
*trainable_variables
+regularization_losses
,	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
-axis
	.gamma
/beta
0moving_mean
1moving_variance
2	variables
3trainable_variables
4regularization_losses
5	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

6kernel
7bias
8	variables
9trainable_variables
:regularization_losses
;	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
<	variables
=trainable_variables
>regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
@axis
	Agamma
Bbeta
Cmoving_mean
Dmoving_variance
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Ikernel
Jbias
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
Saxis
	Tgamma
Ubeta
Vmoving_mean
Wmoving_variance
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

\kernel
]bias
^	variables
_trainable_variables
`regularization_losses
a	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
faxis
	ggamma
hbeta
imoving_mean
jmoving_variance
k	variables
ltrainable_variables
mregularization_losses
n	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
"
_tf_keras_input_layer
?

okernel
pbias
q	variables
rtrainable_variables
sregularization_losses
t	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

ukernel
vbias
w	variables
xtrainable_variables
yregularization_losses
z	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
{	variables
|trainable_variables
}regularization_losses
~	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	?iter
?beta_1
?beta_2

?decay#m?$m?.m?/m?6m?7m?Am?Bm?Im?Jm?Tm?Um?\m?]m?gm?hm?om?pm?um?vm?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?#v?$v?.v?/v?6v?7v?Av?Bv?Iv?Jv?Tv?Uv?\v?]v?gv?hv?ov?pv?uv?vv?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?"
	optimizer
?
#0
$1
.2
/3
04
15
66
77
A8
B9
C10
D11
I12
J13
T14
U15
V16
W17
\18
]19
g20
h21
i22
j23
o24
p25
u26
v27
28
?29
?30
?31
?32
?33
?34
?35
?36
?37
?38
?39
?40
?41
?42
?43
?44
?45"
trackable_list_wrapper
?
#0
$1
.2
/3
64
75
A6
B7
I8
J9
T10
U11
\12
]13
g14
h15
o16
p17
u18
v19
20
?21
?22
?23
?24
?25
?26
?27
?28
?29
?30
?31
?32
?33
?34
?35"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
	variables
 ?layer_regularization_losses
trainable_variables
?layers
?metrics
 regularization_losses
?non_trainable_variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
-:+ 2conv3d_5/kernel
: 2conv3d_5/bias
.
#0
$1"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
%	variables
 ?layer_regularization_losses
&trainable_variables
?layers
?metrics
'regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
)	variables
 ?layer_regularization_losses
*trainable_variables
?layers
?metrics
+regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):' 2batch_normalization_5/gamma
(:& 2batch_normalization_5/beta
1:/  (2!batch_normalization_5/moving_mean
5:3  (2%batch_normalization_5/moving_variance
<
.0
/1
02
13"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
2	variables
 ?layer_regularization_losses
3trainable_variables
?layers
?metrics
4regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-:+  2conv3d_6/kernel
: 2conv3d_6/bias
.
60
71"
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
8	variables
 ?layer_regularization_losses
9trainable_variables
?layers
?metrics
:regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
<	variables
 ?layer_regularization_losses
=trainable_variables
?layers
?metrics
>regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):' 2batch_normalization_6/gamma
(:& 2batch_normalization_6/beta
1:/  (2!batch_normalization_6/moving_mean
5:3  (2%batch_normalization_6/moving_variance
<
A0
B1
C2
D3"
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
E	variables
 ?layer_regularization_losses
Ftrainable_variables
?layers
?metrics
Gregularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-:+ @2conv3d_7/kernel
:@2conv3d_7/bias
.
I0
J1"
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
K	variables
 ?layer_regularization_losses
Ltrainable_variables
?layers
?metrics
Mregularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
O	variables
 ?layer_regularization_losses
Ptrainable_variables
?layers
?metrics
Qregularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'@2batch_normalization_7/gamma
(:&@2batch_normalization_7/beta
1:/@ (2!batch_normalization_7/moving_mean
5:3@ (2%batch_normalization_7/moving_variance
<
T0
U1
V2
W3"
trackable_list_wrapper
.
T0
U1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
X	variables
 ?layer_regularization_losses
Ytrainable_variables
?layers
?metrics
Zregularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.:,@?2conv3d_8/kernel
:?2conv3d_8/bias
.
\0
]1"
trackable_list_wrapper
.
\0
]1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
^	variables
 ?layer_regularization_losses
_trainable_variables
?layers
?metrics
`regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
b	variables
 ?layer_regularization_losses
ctrainable_variables
?layers
?metrics
dregularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(?2batch_normalization_8/gamma
):'?2batch_normalization_8/beta
2:0? (2!batch_normalization_8/moving_mean
6:4? (2%batch_normalization_8/moving_variance
<
g0
h1
i2
j3"
trackable_list_wrapper
.
g0
h1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
k	variables
 ?layer_regularization_losses
ltrainable_variables
?layers
?metrics
mregularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
/:-??2conv3d_9/kernel
:?2conv3d_9/bias
.
o0
p1"
trackable_list_wrapper
.
o0
p1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
q	variables
 ?layer_regularization_losses
rtrainable_variables
?layers
?metrics
sregularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 
21_dense18/kernel
:21_dense18/bias
.
u0
v1"
trackable_list_wrapper
.
u0
v1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
w	variables
 ?layer_regularization_losses
xtrainable_variables
?layers
?metrics
yregularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
{	variables
 ?layer_regularization_losses
|trainable_variables
?layers
?metrics
}regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
":  22_dense32/kernel
: 22_dense32/bias
/
0
?1"
trackable_list_wrapper
/
0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?	variables
 ?layer_regularization_losses
?trainable_variables
?layers
?metrics
?regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(?2batch_normalization_9/gamma
):'?2batch_normalization_9/beta
2:0? (2!batch_normalization_9/moving_mean
6:4? (2%batch_normalization_9/moving_variance
@
?0
?1
?2
?3"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?	variables
 ?layer_regularization_losses
?trainable_variables
?layers
?metrics
?regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
":  @23_dense64/kernel
:@23_dense64/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?	variables
 ?layer_regularization_losses
?trainable_variables
?layers
?metrics
?regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?	variables
 ?layer_regularization_losses
?trainable_variables
?layers
?metrics
?regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": @ 24_dense32/kernel
: 24_dense32/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?	variables
 ?layer_regularization_losses
?trainable_variables
?layers
?metrics
?regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 
??2dense_1/kernel
:?2dense_1/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?	variables
 ?layer_regularization_losses
?trainable_variables
?layers
?metrics
?regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
":  25_dense18/kernel
:25_dense18/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?	variables
 ?layer_regularization_losses
?trainable_variables
?layers
?metrics
?regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?	variables
 ?layer_regularization_losses
?trainable_variables
?layers
?metrics
?regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?	variables
 ?layer_regularization_losses
?trainable_variables
?layers
?metrics
?regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
(:&	?	2combined_input/kernel
!:	2combined_input/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?	variables
 ?layer_regularization_losses
?trainable_variables
?layers
?metrics
?regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
(:&	2ensemble_output/kernel
": 2ensemble_output/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?	variables
 ?layer_regularization_losses
?trainable_variables
?layers
?metrics
?regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
h
00
11
C2
D3
V4
W5
i6
j7
?8
?9"
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
.
00
11"
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
.
C0
D1"
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
.
V0
W1"
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
.
i0
j1"
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
R

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
2:0 2Adam/conv3d_5/kernel/m
 : 2Adam/conv3d_5/bias/m
.:, 2"Adam/batch_normalization_5/gamma/m
-:+ 2!Adam/batch_normalization_5/beta/m
2:0  2Adam/conv3d_6/kernel/m
 : 2Adam/conv3d_6/bias/m
.:, 2"Adam/batch_normalization_6/gamma/m
-:+ 2!Adam/batch_normalization_6/beta/m
2:0 @2Adam/conv3d_7/kernel/m
 :@2Adam/conv3d_7/bias/m
.:,@2"Adam/batch_normalization_7/gamma/m
-:+@2!Adam/batch_normalization_7/beta/m
3:1@?2Adam/conv3d_8/kernel/m
!:?2Adam/conv3d_8/bias/m
/:-?2"Adam/batch_normalization_8/gamma/m
.:,?2!Adam/batch_normalization_8/beta/m
4:2??2Adam/conv3d_9/kernel/m
!:?2Adam/conv3d_9/bias/m
':%
2Adam/1_dense18/kernel/m
!:2Adam/1_dense18/bias/m
':% 2Adam/2_dense32/kernel/m
!: 2Adam/2_dense32/bias/m
/:-?2"Adam/batch_normalization_9/gamma/m
.:,?2!Adam/batch_normalization_9/beta/m
':% @2Adam/3_dense64/kernel/m
!:@2Adam/3_dense64/bias/m
':%@ 2Adam/4_dense32/kernel/m
!: 2Adam/4_dense32/bias/m
':%
??2Adam/dense_1/kernel/m
 :?2Adam/dense_1/bias/m
':% 2Adam/5_dense18/kernel/m
!:2Adam/5_dense18/bias/m
-:+	?	2Adam/combined_input/kernel/m
&:$	2Adam/combined_input/bias/m
-:+	2Adam/ensemble_output/kernel/m
':%2Adam/ensemble_output/bias/m
2:0 2Adam/conv3d_5/kernel/v
 : 2Adam/conv3d_5/bias/v
.:, 2"Adam/batch_normalization_5/gamma/v
-:+ 2!Adam/batch_normalization_5/beta/v
2:0  2Adam/conv3d_6/kernel/v
 : 2Adam/conv3d_6/bias/v
.:, 2"Adam/batch_normalization_6/gamma/v
-:+ 2!Adam/batch_normalization_6/beta/v
2:0 @2Adam/conv3d_7/kernel/v
 :@2Adam/conv3d_7/bias/v
.:,@2"Adam/batch_normalization_7/gamma/v
-:+@2!Adam/batch_normalization_7/beta/v
3:1@?2Adam/conv3d_8/kernel/v
!:?2Adam/conv3d_8/bias/v
/:-?2"Adam/batch_normalization_8/gamma/v
.:,?2!Adam/batch_normalization_8/beta/v
4:2??2Adam/conv3d_9/kernel/v
!:?2Adam/conv3d_9/bias/v
':%
2Adam/1_dense18/kernel/v
!:2Adam/1_dense18/bias/v
':% 2Adam/2_dense32/kernel/v
!: 2Adam/2_dense32/bias/v
/:-?2"Adam/batch_normalization_9/gamma/v
.:,?2!Adam/batch_normalization_9/beta/v
':% @2Adam/3_dense64/kernel/v
!:@2Adam/3_dense64/bias/v
':%@ 2Adam/4_dense32/kernel/v
!: 2Adam/4_dense32/bias/v
':%
??2Adam/dense_1/kernel/v
 :?2Adam/dense_1/bias/v
':% 2Adam/5_dense18/kernel/v
!:2Adam/5_dense18/bias/v
-:+	?	2Adam/combined_input/kernel/v
&:$	2Adam/combined_input/bias/v
-:+	2Adam/ensemble_output/kernel/v
':%2Adam/ensemble_output/bias/v
?B?
"__inference__wrapped_model_9254571
svm_inputsimg3d_inputs"?
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
?2?
,__inference_ensemble4d_layer_call_fn_9255849
,__inference_ensemble4d_layer_call_fn_9257124
,__inference_ensemble4d_layer_call_fn_9257222
,__inference_ensemble4d_layer_call_fn_9256676?
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
G__inference_ensemble4d_layer_call_and_return_conditional_losses_9257398
G__inference_ensemble4d_layer_call_and_return_conditional_losses_9257581
G__inference_ensemble4d_layer_call_and_return_conditional_losses_9256799
G__inference_ensemble4d_layer_call_and_return_conditional_losses_9256922?
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
?2?
*__inference_conv3d_5_layer_call_fn_9257590?
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
E__inference_conv3d_5_layer_call_and_return_conditional_losses_9257601?
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
?2?
1__inference_max_pooling3d_5_layer_call_fn_9257606
1__inference_max_pooling3d_5_layer_call_fn_9257611?
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
?2?
L__inference_max_pooling3d_5_layer_call_and_return_conditional_losses_9257616
L__inference_max_pooling3d_5_layer_call_and_return_conditional_losses_9257621?
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
?2?
7__inference_batch_normalization_5_layer_call_fn_9257634
7__inference_batch_normalization_5_layer_call_fn_9257647
7__inference_batch_normalization_5_layer_call_fn_9257660
7__inference_batch_normalization_5_layer_call_fn_9257673?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_9257691
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_9257709
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_9257727
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_9257745?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
*__inference_conv3d_6_layer_call_fn_9257754?
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
E__inference_conv3d_6_layer_call_and_return_conditional_losses_9257765?
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
?2?
1__inference_max_pooling3d_6_layer_call_fn_9257770
1__inference_max_pooling3d_6_layer_call_fn_9257775?
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
?2?
L__inference_max_pooling3d_6_layer_call_and_return_conditional_losses_9257780
L__inference_max_pooling3d_6_layer_call_and_return_conditional_losses_9257785?
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
?2?
7__inference_batch_normalization_6_layer_call_fn_9257798
7__inference_batch_normalization_6_layer_call_fn_9257811
7__inference_batch_normalization_6_layer_call_fn_9257824
7__inference_batch_normalization_6_layer_call_fn_9257837?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_9257855
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_9257873
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_9257891
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_9257909?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
*__inference_conv3d_7_layer_call_fn_9257918?
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
E__inference_conv3d_7_layer_call_and_return_conditional_losses_9257929?
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
?2?
1__inference_max_pooling3d_7_layer_call_fn_9257934
1__inference_max_pooling3d_7_layer_call_fn_9257939?
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
?2?
L__inference_max_pooling3d_7_layer_call_and_return_conditional_losses_9257944
L__inference_max_pooling3d_7_layer_call_and_return_conditional_losses_9257949?
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
?2?
7__inference_batch_normalization_7_layer_call_fn_9257962
7__inference_batch_normalization_7_layer_call_fn_9257975
7__inference_batch_normalization_7_layer_call_fn_9257988
7__inference_batch_normalization_7_layer_call_fn_9258001?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_9258019
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_9258037
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_9258055
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_9258073?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
*__inference_conv3d_8_layer_call_fn_9258082?
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
E__inference_conv3d_8_layer_call_and_return_conditional_losses_9258093?
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
?2?
1__inference_max_pooling3d_8_layer_call_fn_9258098
1__inference_max_pooling3d_8_layer_call_fn_9258103?
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
?2?
L__inference_max_pooling3d_8_layer_call_and_return_conditional_losses_9258108
L__inference_max_pooling3d_8_layer_call_and_return_conditional_losses_9258113?
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
?2?
7__inference_batch_normalization_8_layer_call_fn_9258126
7__inference_batch_normalization_8_layer_call_fn_9258139
7__inference_batch_normalization_8_layer_call_fn_9258152
7__inference_batch_normalization_8_layer_call_fn_9258165?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_9258183
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_9258201
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_9258219
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_9258237?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
*__inference_conv3d_9_layer_call_fn_9258246?
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
E__inference_conv3d_9_layer_call_and_return_conditional_losses_9258257?
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
+__inference_1_dense18_layer_call_fn_9258266?
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
F__inference_1_dense18_layer_call_and_return_conditional_losses_9258277?
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
?2?
1__inference_max_pooling3d_9_layer_call_fn_9258282
1__inference_max_pooling3d_9_layer_call_fn_9258287?
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
?2?
L__inference_max_pooling3d_9_layer_call_and_return_conditional_losses_9258292
L__inference_max_pooling3d_9_layer_call_and_return_conditional_losses_9258297?
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
+__inference_2_dense32_layer_call_fn_9258306?
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
F__inference_2_dense32_layer_call_and_return_conditional_losses_9258317?
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
?2?
7__inference_batch_normalization_9_layer_call_fn_9258330
7__inference_batch_normalization_9_layer_call_fn_9258343
7__inference_batch_normalization_9_layer_call_fn_9258356
7__inference_batch_normalization_9_layer_call_fn_9258369?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_9258387
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_9258405
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_9258423
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_9258441?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
+__inference_3_dense64_layer_call_fn_9258450?
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
F__inference_3_dense64_layer_call_and_return_conditional_losses_9258461?
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
?2?
<__inference_global_average_pooling3d_1_layer_call_fn_9258466
<__inference_global_average_pooling3d_1_layer_call_fn_9258471?
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
?2?
W__inference_global_average_pooling3d_1_layer_call_and_return_conditional_losses_9258477
W__inference_global_average_pooling3d_1_layer_call_and_return_conditional_losses_9258483?
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
+__inference_4_dense32_layer_call_fn_9258492?
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
F__inference_4_dense32_layer_call_and_return_conditional_losses_9258503?
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
)__inference_dense_1_layer_call_fn_9258512?
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
D__inference_dense_1_layer_call_and_return_conditional_losses_9258523?
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
+__inference_5_dense18_layer_call_fn_9258532?
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
F__inference_5_dense18_layer_call_and_return_conditional_losses_9258543?
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
?2?
+__inference_dropout_1_layer_call_fn_9258548
+__inference_dropout_1_layer_call_fn_9258553?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_dropout_1_layer_call_and_return_conditional_losses_9258558
F__inference_dropout_1_layer_call_and_return_conditional_losses_9258570?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
/__inference_concatenate_1_layer_call_fn_9258576?
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
J__inference_concatenate_1_layer_call_and_return_conditional_losses_9258583?
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
0__inference_combined_input_layer_call_fn_9258592?
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
K__inference_combined_input_layer_call_and_return_conditional_losses_9258603?
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
1__inference_ensemble_output_layer_call_fn_9258612?
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
L__inference_ensemble_output_layer_call_and_return_conditional_losses_9258623?
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
%__inference_signature_wrapper_9257026img3d_inputs
svm_inputs"?
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
F__inference_1_dense18_layer_call_and_return_conditional_losses_9258277\uv/?,
%?"
 ?
inputs?????????

? "%?"
?
0?????????
? ~
+__inference_1_dense18_layer_call_fn_9258266Ouv/?,
%?"
 ?
inputs?????????

? "???????????
F__inference_2_dense32_layer_call_and_return_conditional_losses_9258317]?/?,
%?"
 ?
inputs?????????
? "%?"
?
0????????? 
? 
+__inference_2_dense32_layer_call_fn_9258306P?/?,
%?"
 ?
inputs?????????
? "?????????? ?
F__inference_3_dense64_layer_call_and_return_conditional_losses_9258461^??/?,
%?"
 ?
inputs????????? 
? "%?"
?
0?????????@
? ?
+__inference_3_dense64_layer_call_fn_9258450Q??/?,
%?"
 ?
inputs????????? 
? "??????????@?
F__inference_4_dense32_layer_call_and_return_conditional_losses_9258503^??/?,
%?"
 ?
inputs?????????@
? "%?"
?
0????????? 
? ?
+__inference_4_dense32_layer_call_fn_9258492Q??/?,
%?"
 ?
inputs?????????@
? "?????????? ?
F__inference_5_dense18_layer_call_and_return_conditional_losses_9258543^??/?,
%?"
 ?
inputs????????? 
? "%?"
?
0?????????
? ?
+__inference_5_dense18_layer_call_fn_9258532Q??/?,
%?"
 ?
inputs????????? 
? "???????????
"__inference__wrapped_model_9254571??#$./0167ABCDIJTUVW\]ghijopuv?????????????????n?k
d?a
_?\
$?!

svm_inputs?????????

4?1
img3d_inputs???????????
? "A?>
<
ensemble_output)?&
ensemble_output??????????
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_9257691?./01Z?W
P?M
G?D
inputs8???????????????????????????????????? 
p 
? "L?I
B??
08???????????????????????????????????? 
? ?
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_9257709?./01Z?W
P?M
G?D
inputs8???????????????????????????????????? 
p
? "L?I
B??
08???????????????????????????????????? 
? ?
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_9257727z./01??<
5?2
,?)
inputs?????????@@ 
p 
? "1?.
'?$
0?????????@@ 
? ?
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_9257745z./01??<
5?2
,?)
inputs?????????@@ 
p
? "1?.
'?$
0?????????@@ 
? ?
7__inference_batch_normalization_5_layer_call_fn_9257634?./01Z?W
P?M
G?D
inputs8???????????????????????????????????? 
p 
? "??<8???????????????????????????????????? ?
7__inference_batch_normalization_5_layer_call_fn_9257647?./01Z?W
P?M
G?D
inputs8???????????????????????????????????? 
p
? "??<8???????????????????????????????????? ?
7__inference_batch_normalization_5_layer_call_fn_9257660m./01??<
5?2
,?)
inputs?????????@@ 
p 
? "$?!?????????@@ ?
7__inference_batch_normalization_5_layer_call_fn_9257673m./01??<
5?2
,?)
inputs?????????@@ 
p
? "$?!?????????@@ ?
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_9257855?ABCDZ?W
P?M
G?D
inputs8???????????????????????????????????? 
p 
? "L?I
B??
08???????????????????????????????????? 
? ?
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_9257873?ABCDZ?W
P?M
G?D
inputs8???????????????????????????????????? 
p
? "L?I
B??
08???????????????????????????????????? 
? ?
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_9257891zABCD??<
5?2
,?)
inputs?????????@@ 
p 
? "1?.
'?$
0?????????@@ 
? ?
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_9257909zABCD??<
5?2
,?)
inputs?????????@@ 
p
? "1?.
'?$
0?????????@@ 
? ?
7__inference_batch_normalization_6_layer_call_fn_9257798?ABCDZ?W
P?M
G?D
inputs8???????????????????????????????????? 
p 
? "??<8???????????????????????????????????? ?
7__inference_batch_normalization_6_layer_call_fn_9257811?ABCDZ?W
P?M
G?D
inputs8???????????????????????????????????? 
p
? "??<8???????????????????????????????????? ?
7__inference_batch_normalization_6_layer_call_fn_9257824mABCD??<
5?2
,?)
inputs?????????@@ 
p 
? "$?!?????????@@ ?
7__inference_batch_normalization_6_layer_call_fn_9257837mABCD??<
5?2
,?)
inputs?????????@@ 
p
? "$?!?????????@@ ?
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_9258019?TUVWZ?W
P?M
G?D
inputs8????????????????????????????????????@
p 
? "L?I
B??
08????????????????????????????????????@
? ?
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_9258037?TUVWZ?W
P?M
G?D
inputs8????????????????????????????????????@
p
? "L?I
B??
08????????????????????????????????????@
? ?
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_9258055zTUVW??<
5?2
,?)
inputs?????????@@@
p 
? "1?.
'?$
0?????????@@@
? ?
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_9258073zTUVW??<
5?2
,?)
inputs?????????@@@
p
? "1?.
'?$
0?????????@@@
? ?
7__inference_batch_normalization_7_layer_call_fn_9257962?TUVWZ?W
P?M
G?D
inputs8????????????????????????????????????@
p 
? "??<8????????????????????????????????????@?
7__inference_batch_normalization_7_layer_call_fn_9257975?TUVWZ?W
P?M
G?D
inputs8????????????????????????????????????@
p
? "??<8????????????????????????????????????@?
7__inference_batch_normalization_7_layer_call_fn_9257988mTUVW??<
5?2
,?)
inputs?????????@@@
p 
? "$?!?????????@@@?
7__inference_batch_normalization_7_layer_call_fn_9258001mTUVW??<
5?2
,?)
inputs?????????@@@
p
? "$?!?????????@@@?
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_9258183?ghij[?X
Q?N
H?E
inputs9?????????????????????????????????????
p 
? "M?J
C?@
09?????????????????????????????????????
? ?
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_9258201?ghij[?X
Q?N
H?E
inputs9?????????????????????????????????????
p
? "M?J
C?@
09?????????????????????????????????????
? ?
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_9258219|ghij@?=
6?3
-?*
inputs?????????@@?
p 
? "2?/
(?%
0?????????@@?
? ?
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_9258237|ghij@?=
6?3
-?*
inputs?????????@@?
p
? "2?/
(?%
0?????????@@?
? ?
7__inference_batch_normalization_8_layer_call_fn_9258126?ghij[?X
Q?N
H?E
inputs9?????????????????????????????????????
p 
? "@?=9??????????????????????????????????????
7__inference_batch_normalization_8_layer_call_fn_9258139?ghij[?X
Q?N
H?E
inputs9?????????????????????????????????????
p
? "@?=9??????????????????????????????????????
7__inference_batch_normalization_8_layer_call_fn_9258152oghij@?=
6?3
-?*
inputs?????????@@?
p 
? "%?"?????????@@??
7__inference_batch_normalization_8_layer_call_fn_9258165oghij@?=
6?3
-?*
inputs?????????@@?
p
? "%?"?????????@@??
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_9258387?????[?X
Q?N
H?E
inputs9?????????????????????????????????????
p 
? "M?J
C?@
09?????????????????????????????????????
? ?
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_9258405?????[?X
Q?N
H?E
inputs9?????????????????????????????????????
p
? "M?J
C?@
09?????????????????????????????????????
? ?
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_9258423?????@?=
6?3
-?*
inputs?????????@@?
p 
? "2?/
(?%
0?????????@@?
? ?
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_9258441?????@?=
6?3
-?*
inputs?????????@@?
p
? "2?/
(?%
0?????????@@?
? ?
7__inference_batch_normalization_9_layer_call_fn_9258330?????[?X
Q?N
H?E
inputs9?????????????????????????????????????
p 
? "@?=9??????????????????????????????????????
7__inference_batch_normalization_9_layer_call_fn_9258343?????[?X
Q?N
H?E
inputs9?????????????????????????????????????
p
? "@?=9??????????????????????????????????????
7__inference_batch_normalization_9_layer_call_fn_9258356s????@?=
6?3
-?*
inputs?????????@@?
p 
? "%?"?????????@@??
7__inference_batch_normalization_9_layer_call_fn_9258369s????@?=
6?3
-?*
inputs?????????@@?
p
? "%?"?????????@@??
K__inference_combined_input_layer_call_and_return_conditional_losses_9258603_??0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????	
? ?
0__inference_combined_input_layer_call_fn_9258592R??0?-
&?#
!?
inputs??????????
? "??????????	?
J__inference_concatenate_1_layer_call_and_return_conditional_losses_9258583?[?X
Q?N
L?I
"?
inputs/0?????????
#? 
inputs/1??????????
? "&?#
?
0??????????
? ?
/__inference_concatenate_1_layer_call_fn_9258576x[?X
Q?N
L?I
"?
inputs/0?????????
#? 
inputs/1??????????
? "????????????
E__inference_conv3d_5_layer_call_and_return_conditional_losses_9257601x#$=?:
3?0
.?+
inputs???????????
? "3?0
)?&
0??????????? 
? ?
*__inference_conv3d_5_layer_call_fn_9257590k#$=?:
3?0
.?+
inputs???????????
? "&?#??????????? ?
E__inference_conv3d_6_layer_call_and_return_conditional_losses_9257765t67;?8
1?.
,?)
inputs?????????@@ 
? "1?.
'?$
0?????????@@ 
? ?
*__inference_conv3d_6_layer_call_fn_9257754g67;?8
1?.
,?)
inputs?????????@@ 
? "$?!?????????@@ ?
E__inference_conv3d_7_layer_call_and_return_conditional_losses_9257929tIJ;?8
1?.
,?)
inputs?????????@@ 
? "1?.
'?$
0?????????@@@
? ?
*__inference_conv3d_7_layer_call_fn_9257918gIJ;?8
1?.
,?)
inputs?????????@@ 
? "$?!?????????@@@?
E__inference_conv3d_8_layer_call_and_return_conditional_losses_9258093u\];?8
1?.
,?)
inputs?????????@@@
? "2?/
(?%
0?????????@@?
? ?
*__inference_conv3d_8_layer_call_fn_9258082h\];?8
1?.
,?)
inputs?????????@@@
? "%?"?????????@@??
E__inference_conv3d_9_layer_call_and_return_conditional_losses_9258257vop<?9
2?/
-?*
inputs?????????@@?
? "2?/
(?%
0?????????@@?
? ?
*__inference_conv3d_9_layer_call_fn_9258246iop<?9
2?/
-?*
inputs?????????@@?
? "%?"?????????@@??
D__inference_dense_1_layer_call_and_return_conditional_losses_9258523`??0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
)__inference_dense_1_layer_call_fn_9258512S??0?-
&?#
!?
inputs??????????
? "????????????
F__inference_dropout_1_layer_call_and_return_conditional_losses_9258558^4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ?
F__inference_dropout_1_layer_call_and_return_conditional_losses_9258570^4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? ?
+__inference_dropout_1_layer_call_fn_9258548Q4?1
*?'
!?
inputs??????????
p 
? "????????????
+__inference_dropout_1_layer_call_fn_9258553Q4?1
*?'
!?
inputs??????????
p
? "????????????
G__inference_ensemble4d_layer_call_and_return_conditional_losses_9256799??#$./0167ABCDIJTUVW\]ghijopuv?????????????????v?s
l?i
_?\
$?!

svm_inputs?????????

4?1
img3d_inputs???????????
p 

 
? "%?"
?
0?????????
? ?
G__inference_ensemble4d_layer_call_and_return_conditional_losses_9256922??#$./0167ABCDIJTUVW\]ghijopuv?????????????????v?s
l?i
_?\
$?!

svm_inputs?????????

4?1
img3d_inputs???????????
p

 
? "%?"
?
0?????????
? ?
G__inference_ensemble4d_layer_call_and_return_conditional_losses_9257398??#$./0167ABCDIJTUVW\]ghijopuv?????????????????p?m
f?c
Y?V
"?
inputs/0?????????

0?-
inputs/1???????????
p 

 
? "%?"
?
0?????????
? ?
G__inference_ensemble4d_layer_call_and_return_conditional_losses_9257581??#$./0167ABCDIJTUVW\]ghijopuv?????????????????p?m
f?c
Y?V
"?
inputs/0?????????

0?-
inputs/1???????????
p

 
? "%?"
?
0?????????
? ?
,__inference_ensemble4d_layer_call_fn_9255849??#$./0167ABCDIJTUVW\]ghijopuv?????????????????v?s
l?i
_?\
$?!

svm_inputs?????????

4?1
img3d_inputs???????????
p 

 
? "???????????
,__inference_ensemble4d_layer_call_fn_9256676??#$./0167ABCDIJTUVW\]ghijopuv?????????????????v?s
l?i
_?\
$?!

svm_inputs?????????

4?1
img3d_inputs???????????
p

 
? "???????????
,__inference_ensemble4d_layer_call_fn_9257124??#$./0167ABCDIJTUVW\]ghijopuv?????????????????p?m
f?c
Y?V
"?
inputs/0?????????

0?-
inputs/1???????????
p 

 
? "???????????
,__inference_ensemble4d_layer_call_fn_9257222??#$./0167ABCDIJTUVW\]ghijopuv?????????????????p?m
f?c
Y?V
"?
inputs/0?????????

0?-
inputs/1???????????
p

 
? "???????????
L__inference_ensemble_output_layer_call_and_return_conditional_losses_9258623^??/?,
%?"
 ?
inputs?????????	
? "%?"
?
0?????????
? ?
1__inference_ensemble_output_layer_call_fn_9258612Q??/?,
%?"
 ?
inputs?????????	
? "???????????
W__inference_global_average_pooling3d_1_layer_call_and_return_conditional_losses_9258477?_?\
U?R
P?M
inputsA?????????????????????????????????????????????
? ".?+
$?!
0??????????????????
? ?
W__inference_global_average_pooling3d_1_layer_call_and_return_conditional_losses_9258483f<?9
2?/
-?*
inputs?????????@@?
? "&?#
?
0??????????
? ?
<__inference_global_average_pooling3d_1_layer_call_fn_9258466?_?\
U?R
P?M
inputsA?????????????????????????????????????????????
? "!????????????????????
<__inference_global_average_pooling3d_1_layer_call_fn_9258471Y<?9
2?/
-?*
inputs?????????@@?
? "????????????
L__inference_max_pooling3d_5_layer_call_and_return_conditional_losses_9257616?_?\
U?R
P?M
inputsA?????????????????????????????????????????????
? "U?R
K?H
0A?????????????????????????????????????????????
? ?
L__inference_max_pooling3d_5_layer_call_and_return_conditional_losses_9257621r=?:
3?0
.?+
inputs??????????? 
? "1?.
'?$
0?????????@@ 
? ?
1__inference_max_pooling3d_5_layer_call_fn_9257606?_?\
U?R
P?M
inputsA?????????????????????????????????????????????
? "H?EA??????????????????????????????????????????????
1__inference_max_pooling3d_5_layer_call_fn_9257611e=?:
3?0
.?+
inputs??????????? 
? "$?!?????????@@ ?
L__inference_max_pooling3d_6_layer_call_and_return_conditional_losses_9257780?_?\
U?R
P?M
inputsA?????????????????????????????????????????????
? "U?R
K?H
0A?????????????????????????????????????????????
? ?
L__inference_max_pooling3d_6_layer_call_and_return_conditional_losses_9257785p;?8
1?.
,?)
inputs?????????@@ 
? "1?.
'?$
0?????????@@ 
? ?
1__inference_max_pooling3d_6_layer_call_fn_9257770?_?\
U?R
P?M
inputsA?????????????????????????????????????????????
? "H?EA??????????????????????????????????????????????
1__inference_max_pooling3d_6_layer_call_fn_9257775c;?8
1?.
,?)
inputs?????????@@ 
? "$?!?????????@@ ?
L__inference_max_pooling3d_7_layer_call_and_return_conditional_losses_9257944?_?\
U?R
P?M
inputsA?????????????????????????????????????????????
? "U?R
K?H
0A?????????????????????????????????????????????
? ?
L__inference_max_pooling3d_7_layer_call_and_return_conditional_losses_9257949p;?8
1?.
,?)
inputs?????????@@@
? "1?.
'?$
0?????????@@@
? ?
1__inference_max_pooling3d_7_layer_call_fn_9257934?_?\
U?R
P?M
inputsA?????????????????????????????????????????????
? "H?EA??????????????????????????????????????????????
1__inference_max_pooling3d_7_layer_call_fn_9257939c;?8
1?.
,?)
inputs?????????@@@
? "$?!?????????@@@?
L__inference_max_pooling3d_8_layer_call_and_return_conditional_losses_9258108?_?\
U?R
P?M
inputsA?????????????????????????????????????????????
? "U?R
K?H
0A?????????????????????????????????????????????
? ?
L__inference_max_pooling3d_8_layer_call_and_return_conditional_losses_9258113r<?9
2?/
-?*
inputs?????????@@?
? "2?/
(?%
0?????????@@?
? ?
1__inference_max_pooling3d_8_layer_call_fn_9258098?_?\
U?R
P?M
inputsA?????????????????????????????????????????????
? "H?EA??????????????????????????????????????????????
1__inference_max_pooling3d_8_layer_call_fn_9258103e<?9
2?/
-?*
inputs?????????@@?
? "%?"?????????@@??
L__inference_max_pooling3d_9_layer_call_and_return_conditional_losses_9258292?_?\
U?R
P?M
inputsA?????????????????????????????????????????????
? "U?R
K?H
0A?????????????????????????????????????????????
? ?
L__inference_max_pooling3d_9_layer_call_and_return_conditional_losses_9258297r<?9
2?/
-?*
inputs?????????@@?
? "2?/
(?%
0?????????@@?
? ?
1__inference_max_pooling3d_9_layer_call_fn_9258282?_?\
U?R
P?M
inputsA?????????????????????????????????????????????
? "H?EA??????????????????????????????????????????????
1__inference_max_pooling3d_9_layer_call_fn_9258287e<?9
2?/
-?*
inputs?????????@@?
? "%?"?????????@@??
%__inference_signature_wrapper_9257026??#$./0167ABCDIJTUVW\]ghijopuv????????????????????
? 
}?z
D
img3d_inputs4?1
img3d_inputs???????????
2

svm_inputs$?!

svm_inputs?????????
"A?>
<
ensemble_output)?&
ensemble_output?????????
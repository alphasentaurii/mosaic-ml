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
conv3d_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv3d_10/kernel
?
$conv3d_10/kernel/Read/ReadVariableOpReadVariableOpconv3d_10/kernel**
_output_shapes
: *
dtype0
t
conv3d_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv3d_10/bias
m
"conv3d_10/bias/Read/ReadVariableOpReadVariableOpconv3d_10/bias*
_output_shapes
: *
dtype0
?
batch_normalization_10/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_10/gamma
?
0batch_normalization_10/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_10/gamma*
_output_shapes
: *
dtype0
?
batch_normalization_10/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_10/beta
?
/batch_normalization_10/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_10/beta*
_output_shapes
: *
dtype0
?
"batch_normalization_10/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"batch_normalization_10/moving_mean
?
6batch_normalization_10/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_10/moving_mean*
_output_shapes
: *
dtype0
?
&batch_normalization_10/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&batch_normalization_10/moving_variance
?
:batch_normalization_10/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_10/moving_variance*
_output_shapes
: *
dtype0
?
conv3d_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *!
shared_nameconv3d_11/kernel
?
$conv3d_11/kernel/Read/ReadVariableOpReadVariableOpconv3d_11/kernel**
_output_shapes
:  *
dtype0
t
conv3d_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv3d_11/bias
m
"conv3d_11/bias/Read/ReadVariableOpReadVariableOpconv3d_11/bias*
_output_shapes
: *
dtype0
?
batch_normalization_11/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_11/gamma
?
0batch_normalization_11/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_11/gamma*
_output_shapes
: *
dtype0
?
batch_normalization_11/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_11/beta
?
/batch_normalization_11/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_11/beta*
_output_shapes
: *
dtype0
?
"batch_normalization_11/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"batch_normalization_11/moving_mean
?
6batch_normalization_11/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_11/moving_mean*
_output_shapes
: *
dtype0
?
&batch_normalization_11/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&batch_normalization_11/moving_variance
?
:batch_normalization_11/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_11/moving_variance*
_output_shapes
: *
dtype0
?
conv3d_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*!
shared_nameconv3d_12/kernel
?
$conv3d_12/kernel/Read/ReadVariableOpReadVariableOpconv3d_12/kernel**
_output_shapes
: @*
dtype0
t
conv3d_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv3d_12/bias
m
"conv3d_12/bias/Read/ReadVariableOpReadVariableOpconv3d_12/bias*
_output_shapes
:@*
dtype0
?
batch_normalization_12/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_12/gamma
?
0batch_normalization_12/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_12/gamma*
_output_shapes
:@*
dtype0
?
batch_normalization_12/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_12/beta
?
/batch_normalization_12/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_12/beta*
_output_shapes
:@*
dtype0
?
"batch_normalization_12/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"batch_normalization_12/moving_mean
?
6batch_normalization_12/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_12/moving_mean*
_output_shapes
:@*
dtype0
?
&batch_normalization_12/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&batch_normalization_12/moving_variance
?
:batch_normalization_12/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_12/moving_variance*
_output_shapes
:@*
dtype0
?
conv3d_13/kernelVarHandleOp*
_output_shapes
: *
dtype0* 
shape:@?*!
shared_nameconv3d_13/kernel
?
$conv3d_13/kernel/Read/ReadVariableOpReadVariableOpconv3d_13/kernel*+
_output_shapes
:@?*
dtype0
u
conv3d_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv3d_13/bias
n
"conv3d_13/bias/Read/ReadVariableOpReadVariableOpconv3d_13/bias*
_output_shapes	
:?*
dtype0
?
batch_normalization_13/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*-
shared_namebatch_normalization_13/gamma
?
0batch_normalization_13/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_13/gamma*
_output_shapes	
:?*
dtype0
?
batch_normalization_13/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_namebatch_normalization_13/beta
?
/batch_normalization_13/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_13/beta*
_output_shapes	
:?*
dtype0
?
"batch_normalization_13/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"batch_normalization_13/moving_mean
?
6batch_normalization_13/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_13/moving_mean*
_output_shapes	
:?*
dtype0
?
&batch_normalization_13/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*7
shared_name(&batch_normalization_13/moving_variance
?
:batch_normalization_13/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_13/moving_variance*
_output_shapes	
:?*
dtype0
?
conv3d_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*!
shape:??*!
shared_nameconv3d_14/kernel
?
$conv3d_14/kernel/Read/ReadVariableOpReadVariableOpconv3d_14/kernel*,
_output_shapes
:??*
dtype0
u
conv3d_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv3d_14/bias
n
"conv3d_14/bias/Read/ReadVariableOpReadVariableOpconv3d_14/bias*
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
batch_normalization_14/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*-
shared_namebatch_normalization_14/gamma
?
0batch_normalization_14/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_14/gamma*
_output_shapes	
:?*
dtype0
?
batch_normalization_14/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_namebatch_normalization_14/beta
?
/batch_normalization_14/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_14/beta*
_output_shapes	
:?*
dtype0
?
"batch_normalization_14/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"batch_normalization_14/moving_mean
?
6batch_normalization_14/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_14/moving_mean*
_output_shapes	
:?*
dtype0
?
&batch_normalization_14/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*7
shared_name(&batch_normalization_14/moving_variance
?
:batch_normalization_14/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_14/moving_variance*
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
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_namedense_2/kernel
s
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel* 
_output_shapes
:
??*
dtype0
q
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_2/bias
j
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
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
Adam/conv3d_10/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv3d_10/kernel/m
?
+Adam/conv3d_10/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_10/kernel/m**
_output_shapes
: *
dtype0
?
Adam/conv3d_10/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv3d_10/bias/m
{
)Adam/conv3d_10/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_10/bias/m*
_output_shapes
: *
dtype0
?
#Adam/batch_normalization_10/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/batch_normalization_10/gamma/m
?
7Adam/batch_normalization_10/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_10/gamma/m*
_output_shapes
: *
dtype0
?
"Adam/batch_normalization_10/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_10/beta/m
?
6Adam/batch_normalization_10/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_10/beta/m*
_output_shapes
: *
dtype0
?
Adam/conv3d_11/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *(
shared_nameAdam/conv3d_11/kernel/m
?
+Adam/conv3d_11/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_11/kernel/m**
_output_shapes
:  *
dtype0
?
Adam/conv3d_11/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv3d_11/bias/m
{
)Adam/conv3d_11/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_11/bias/m*
_output_shapes
: *
dtype0
?
#Adam/batch_normalization_11/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/batch_normalization_11/gamma/m
?
7Adam/batch_normalization_11/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_11/gamma/m*
_output_shapes
: *
dtype0
?
"Adam/batch_normalization_11/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_11/beta/m
?
6Adam/batch_normalization_11/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_11/beta/m*
_output_shapes
: *
dtype0
?
Adam/conv3d_12/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*(
shared_nameAdam/conv3d_12/kernel/m
?
+Adam/conv3d_12/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_12/kernel/m**
_output_shapes
: @*
dtype0
?
Adam/conv3d_12/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv3d_12/bias/m
{
)Adam/conv3d_12/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_12/bias/m*
_output_shapes
:@*
dtype0
?
#Adam/batch_normalization_12/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/batch_normalization_12/gamma/m
?
7Adam/batch_normalization_12/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_12/gamma/m*
_output_shapes
:@*
dtype0
?
"Adam/batch_normalization_12/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_12/beta/m
?
6Adam/batch_normalization_12/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_12/beta/m*
_output_shapes
:@*
dtype0
?
Adam/conv3d_13/kernel/mVarHandleOp*
_output_shapes
: *
dtype0* 
shape:@?*(
shared_nameAdam/conv3d_13/kernel/m
?
+Adam/conv3d_13/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_13/kernel/m*+
_output_shapes
:@?*
dtype0
?
Adam/conv3d_13/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/conv3d_13/bias/m
|
)Adam/conv3d_13/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_13/bias/m*
_output_shapes	
:?*
dtype0
?
#Adam/batch_normalization_13/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*4
shared_name%#Adam/batch_normalization_13/gamma/m
?
7Adam/batch_normalization_13/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_13/gamma/m*
_output_shapes	
:?*
dtype0
?
"Adam/batch_normalization_13/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"Adam/batch_normalization_13/beta/m
?
6Adam/batch_normalization_13/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_13/beta/m*
_output_shapes	
:?*
dtype0
?
Adam/conv3d_14/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*!
shape:??*(
shared_nameAdam/conv3d_14/kernel/m
?
+Adam/conv3d_14/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_14/kernel/m*,
_output_shapes
:??*
dtype0
?
Adam/conv3d_14/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/conv3d_14/bias/m
|
)Adam/conv3d_14/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_14/bias/m*
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
#Adam/batch_normalization_14/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*4
shared_name%#Adam/batch_normalization_14/gamma/m
?
7Adam/batch_normalization_14/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_14/gamma/m*
_output_shapes	
:?*
dtype0
?
"Adam/batch_normalization_14/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"Adam/batch_normalization_14/beta/m
?
6Adam/batch_normalization_14/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_14/beta/m*
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
Adam/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*&
shared_nameAdam/dense_2/kernel/m
?
)Adam/dense_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/m* 
_output_shapes
:
??*
dtype0

Adam/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*$
shared_nameAdam/dense_2/bias/m
x
'Adam/dense_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/m*
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
Adam/conv3d_10/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv3d_10/kernel/v
?
+Adam/conv3d_10/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_10/kernel/v**
_output_shapes
: *
dtype0
?
Adam/conv3d_10/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv3d_10/bias/v
{
)Adam/conv3d_10/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_10/bias/v*
_output_shapes
: *
dtype0
?
#Adam/batch_normalization_10/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/batch_normalization_10/gamma/v
?
7Adam/batch_normalization_10/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_10/gamma/v*
_output_shapes
: *
dtype0
?
"Adam/batch_normalization_10/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_10/beta/v
?
6Adam/batch_normalization_10/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_10/beta/v*
_output_shapes
: *
dtype0
?
Adam/conv3d_11/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *(
shared_nameAdam/conv3d_11/kernel/v
?
+Adam/conv3d_11/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_11/kernel/v**
_output_shapes
:  *
dtype0
?
Adam/conv3d_11/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv3d_11/bias/v
{
)Adam/conv3d_11/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_11/bias/v*
_output_shapes
: *
dtype0
?
#Adam/batch_normalization_11/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/batch_normalization_11/gamma/v
?
7Adam/batch_normalization_11/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_11/gamma/v*
_output_shapes
: *
dtype0
?
"Adam/batch_normalization_11/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_11/beta/v
?
6Adam/batch_normalization_11/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_11/beta/v*
_output_shapes
: *
dtype0
?
Adam/conv3d_12/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*(
shared_nameAdam/conv3d_12/kernel/v
?
+Adam/conv3d_12/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_12/kernel/v**
_output_shapes
: @*
dtype0
?
Adam/conv3d_12/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv3d_12/bias/v
{
)Adam/conv3d_12/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_12/bias/v*
_output_shapes
:@*
dtype0
?
#Adam/batch_normalization_12/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/batch_normalization_12/gamma/v
?
7Adam/batch_normalization_12/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_12/gamma/v*
_output_shapes
:@*
dtype0
?
"Adam/batch_normalization_12/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_12/beta/v
?
6Adam/batch_normalization_12/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_12/beta/v*
_output_shapes
:@*
dtype0
?
Adam/conv3d_13/kernel/vVarHandleOp*
_output_shapes
: *
dtype0* 
shape:@?*(
shared_nameAdam/conv3d_13/kernel/v
?
+Adam/conv3d_13/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_13/kernel/v*+
_output_shapes
:@?*
dtype0
?
Adam/conv3d_13/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/conv3d_13/bias/v
|
)Adam/conv3d_13/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_13/bias/v*
_output_shapes	
:?*
dtype0
?
#Adam/batch_normalization_13/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*4
shared_name%#Adam/batch_normalization_13/gamma/v
?
7Adam/batch_normalization_13/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_13/gamma/v*
_output_shapes	
:?*
dtype0
?
"Adam/batch_normalization_13/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"Adam/batch_normalization_13/beta/v
?
6Adam/batch_normalization_13/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_13/beta/v*
_output_shapes	
:?*
dtype0
?
Adam/conv3d_14/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*!
shape:??*(
shared_nameAdam/conv3d_14/kernel/v
?
+Adam/conv3d_14/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_14/kernel/v*,
_output_shapes
:??*
dtype0
?
Adam/conv3d_14/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/conv3d_14/bias/v
|
)Adam/conv3d_14/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_14/bias/v*
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
#Adam/batch_normalization_14/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*4
shared_name%#Adam/batch_normalization_14/gamma/v
?
7Adam/batch_normalization_14/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_14/gamma/v*
_output_shapes	
:?*
dtype0
?
"Adam/batch_normalization_14/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"Adam/batch_normalization_14/beta/v
?
6Adam/batch_normalization_14/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_14/beta/v*
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
Adam/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*&
shared_nameAdam/dense_2/kernel/v
?
)Adam/dense_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/v* 
_output_shapes
:
??*
dtype0

Adam/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*$
shared_nameAdam/dense_2/bias/v
x
'Adam/dense_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/v*
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
 ?layer_regularization_losses
	variables
?non_trainable_variables
trainable_variables
?metrics
 regularization_losses
?layer_metrics
?layers
 
\Z
VARIABLE_VALUEconv3d_10/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv3d_10/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

#0
$1

#0
$1
 
?
 ?layer_regularization_losses
%	variables
?non_trainable_variables
&trainable_variables
?metrics
'regularization_losses
?layer_metrics
?layers
 
 
 
?
 ?layer_regularization_losses
)	variables
?non_trainable_variables
*trainable_variables
?metrics
+regularization_losses
?layer_metrics
?layers
 
ge
VARIABLE_VALUEbatch_normalization_10/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_10/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_10/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_10/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

.0
/1
02
13

.0
/1
 
?
 ?layer_regularization_losses
2	variables
?non_trainable_variables
3trainable_variables
?metrics
4regularization_losses
?layer_metrics
?layers
\Z
VARIABLE_VALUEconv3d_11/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv3d_11/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

60
71

60
71
 
?
 ?layer_regularization_losses
8	variables
?non_trainable_variables
9trainable_variables
?metrics
:regularization_losses
?layer_metrics
?layers
 
 
 
?
 ?layer_regularization_losses
<	variables
?non_trainable_variables
=trainable_variables
?metrics
>regularization_losses
?layer_metrics
?layers
 
ge
VARIABLE_VALUEbatch_normalization_11/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_11/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_11/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_11/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

A0
B1
C2
D3

A0
B1
 
?
 ?layer_regularization_losses
E	variables
?non_trainable_variables
Ftrainable_variables
?metrics
Gregularization_losses
?layer_metrics
?layers
\Z
VARIABLE_VALUEconv3d_12/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv3d_12/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

I0
J1

I0
J1
 
?
 ?layer_regularization_losses
K	variables
?non_trainable_variables
Ltrainable_variables
?metrics
Mregularization_losses
?layer_metrics
?layers
 
 
 
?
 ?layer_regularization_losses
O	variables
?non_trainable_variables
Ptrainable_variables
?metrics
Qregularization_losses
?layer_metrics
?layers
 
ge
VARIABLE_VALUEbatch_normalization_12/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_12/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_12/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_12/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

T0
U1
V2
W3

T0
U1
 
?
 ?layer_regularization_losses
X	variables
?non_trainable_variables
Ytrainable_variables
?metrics
Zregularization_losses
?layer_metrics
?layers
\Z
VARIABLE_VALUEconv3d_13/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv3d_13/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

\0
]1

\0
]1
 
?
 ?layer_regularization_losses
^	variables
?non_trainable_variables
_trainable_variables
?metrics
`regularization_losses
?layer_metrics
?layers
 
 
 
?
 ?layer_regularization_losses
b	variables
?non_trainable_variables
ctrainable_variables
?metrics
dregularization_losses
?layer_metrics
?layers
 
ge
VARIABLE_VALUEbatch_normalization_13/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_13/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_13/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_13/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

g0
h1
i2
j3

g0
h1
 
?
 ?layer_regularization_losses
k	variables
?non_trainable_variables
ltrainable_variables
?metrics
mregularization_losses
?layer_metrics
?layers
\Z
VARIABLE_VALUEconv3d_14/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv3d_14/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

o0
p1

o0
p1
 
?
 ?layer_regularization_losses
q	variables
?non_trainable_variables
rtrainable_variables
?metrics
sregularization_losses
?layer_metrics
?layers
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
 ?layer_regularization_losses
w	variables
?non_trainable_variables
xtrainable_variables
?metrics
yregularization_losses
?layer_metrics
?layers
 
 
 
?
 ?layer_regularization_losses
{	variables
?non_trainable_variables
|trainable_variables
?metrics
}regularization_losses
?layer_metrics
?layers
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
 ?layer_regularization_losses
?	variables
?non_trainable_variables
?trainable_variables
?metrics
?regularization_losses
?layer_metrics
?layers
 
hf
VARIABLE_VALUEbatch_normalization_14/gamma6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEbatch_normalization_14/beta5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE"batch_normalization_14/moving_mean<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE&batch_normalization_14/moving_variance@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
?0
?1
?2
?3

?0
?1
 
?
 ?layer_regularization_losses
?	variables
?non_trainable_variables
?trainable_variables
?metrics
?regularization_losses
?layer_metrics
?layers
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
 ?layer_regularization_losses
?	variables
?non_trainable_variables
?trainable_variables
?metrics
?regularization_losses
?layer_metrics
?layers
 
 
 
?
 ?layer_regularization_losses
?	variables
?non_trainable_variables
?trainable_variables
?metrics
?regularization_losses
?layer_metrics
?layers
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
 ?layer_regularization_losses
?	variables
?non_trainable_variables
?trainable_variables
?metrics
?regularization_losses
?layer_metrics
?layers
[Y
VARIABLE_VALUEdense_2/kernel7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_2/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?0
?1
 
?
 ?layer_regularization_losses
?	variables
?non_trainable_variables
?trainable_variables
?metrics
?regularization_losses
?layer_metrics
?layers
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
 ?layer_regularization_losses
?	variables
?non_trainable_variables
?trainable_variables
?metrics
?regularization_losses
?layer_metrics
?layers
 
 
 
?
 ?layer_regularization_losses
?	variables
?non_trainable_variables
?trainable_variables
?metrics
?regularization_losses
?layer_metrics
?layers
 
 
 
?
 ?layer_regularization_losses
?	variables
?non_trainable_variables
?trainable_variables
?metrics
?regularization_losses
?layer_metrics
?layers
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
 ?layer_regularization_losses
?	variables
?non_trainable_variables
?trainable_variables
?metrics
?regularization_losses
?layer_metrics
?layers
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
 ?layer_regularization_losses
?	variables
?non_trainable_variables
?trainable_variables
?metrics
?regularization_losses
?layer_metrics
?layers
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

?0
?1
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
}
VARIABLE_VALUEAdam/conv3d_10/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv3d_10/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_10/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_10/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv3d_11/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv3d_11/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_11/gamma/mQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_11/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv3d_12/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv3d_12/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_12/gamma/mQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_12/beta/mPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv3d_13/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv3d_13/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_13/gamma/mQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_13/beta/mPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv3d_14/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv3d_14/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/1_dense18/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/1_dense18/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/2_dense32/kernel/mSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/2_dense32/bias/mQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_14/gamma/mRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_14/beta/mQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/3_dense64/kernel/mSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/3_dense64/bias/mQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/4_dense32/kernel/mSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/4_dense32/bias/mQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_2/kernel/mSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_2/bias/mQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
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
}
VARIABLE_VALUEAdam/conv3d_10/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv3d_10/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_10/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_10/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv3d_11/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv3d_11/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_11/gamma/vQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_11/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv3d_12/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv3d_12/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_12/gamma/vQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_12/beta/vPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv3d_13/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv3d_13/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_13/gamma/vQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_13/beta/vPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv3d_14/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv3d_14/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/1_dense18/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/1_dense18/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/2_dense32/kernel/vSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/2_dense32/bias/vQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_14/gamma/vRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_14/beta/vQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/3_dense64/kernel/vSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/3_dense64/bias/vQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/4_dense32/kernel/vSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/4_dense32/bias/vQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_2/kernel/vSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_2/bias/vQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
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
StatefulPartitionedCallStatefulPartitionedCallserving_default_img3d_inputsserving_default_svm_inputsconv3d_10/kernelconv3d_10/biasbatch_normalization_10/gammabatch_normalization_10/beta"batch_normalization_10/moving_mean&batch_normalization_10/moving_varianceconv3d_11/kernelconv3d_11/biasbatch_normalization_11/gammabatch_normalization_11/beta"batch_normalization_11/moving_mean&batch_normalization_11/moving_varianceconv3d_12/kernelconv3d_12/biasbatch_normalization_12/gammabatch_normalization_12/beta"batch_normalization_12/moving_mean&batch_normalization_12/moving_varianceconv3d_13/kernelconv3d_13/biasbatch_normalization_13/gammabatch_normalization_13/beta"batch_normalization_13/moving_mean&batch_normalization_13/moving_varianceconv3d_14/kernelconv3d_14/bias1_dense18/kernel1_dense18/biasbatch_normalization_14/gammabatch_normalization_14/beta"batch_normalization_14/moving_mean&batch_normalization_14/moving_variance2_dense32/kernel2_dense32/bias3_dense64/kernel3_dense64/biasdense_2/kerneldense_2/bias4_dense32/kernel4_dense32/bias5_dense18/kernel5_dense18/biascombined_input/kernelcombined_input/biasensemble_output/kernelensemble_output/bias*;
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
%__inference_signature_wrapper_9335780
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?0
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv3d_10/kernel/Read/ReadVariableOp"conv3d_10/bias/Read/ReadVariableOp0batch_normalization_10/gamma/Read/ReadVariableOp/batch_normalization_10/beta/Read/ReadVariableOp6batch_normalization_10/moving_mean/Read/ReadVariableOp:batch_normalization_10/moving_variance/Read/ReadVariableOp$conv3d_11/kernel/Read/ReadVariableOp"conv3d_11/bias/Read/ReadVariableOp0batch_normalization_11/gamma/Read/ReadVariableOp/batch_normalization_11/beta/Read/ReadVariableOp6batch_normalization_11/moving_mean/Read/ReadVariableOp:batch_normalization_11/moving_variance/Read/ReadVariableOp$conv3d_12/kernel/Read/ReadVariableOp"conv3d_12/bias/Read/ReadVariableOp0batch_normalization_12/gamma/Read/ReadVariableOp/batch_normalization_12/beta/Read/ReadVariableOp6batch_normalization_12/moving_mean/Read/ReadVariableOp:batch_normalization_12/moving_variance/Read/ReadVariableOp$conv3d_13/kernel/Read/ReadVariableOp"conv3d_13/bias/Read/ReadVariableOp0batch_normalization_13/gamma/Read/ReadVariableOp/batch_normalization_13/beta/Read/ReadVariableOp6batch_normalization_13/moving_mean/Read/ReadVariableOp:batch_normalization_13/moving_variance/Read/ReadVariableOp$conv3d_14/kernel/Read/ReadVariableOp"conv3d_14/bias/Read/ReadVariableOp$1_dense18/kernel/Read/ReadVariableOp"1_dense18/bias/Read/ReadVariableOp$2_dense32/kernel/Read/ReadVariableOp"2_dense32/bias/Read/ReadVariableOp0batch_normalization_14/gamma/Read/ReadVariableOp/batch_normalization_14/beta/Read/ReadVariableOp6batch_normalization_14/moving_mean/Read/ReadVariableOp:batch_normalization_14/moving_variance/Read/ReadVariableOp$3_dense64/kernel/Read/ReadVariableOp"3_dense64/bias/Read/ReadVariableOp$4_dense32/kernel/Read/ReadVariableOp"4_dense32/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp$5_dense18/kernel/Read/ReadVariableOp"5_dense18/bias/Read/ReadVariableOp)combined_input/kernel/Read/ReadVariableOp'combined_input/bias/Read/ReadVariableOp*ensemble_output/kernel/Read/ReadVariableOp(ensemble_output/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp+Adam/conv3d_10/kernel/m/Read/ReadVariableOp)Adam/conv3d_10/bias/m/Read/ReadVariableOp7Adam/batch_normalization_10/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_10/beta/m/Read/ReadVariableOp+Adam/conv3d_11/kernel/m/Read/ReadVariableOp)Adam/conv3d_11/bias/m/Read/ReadVariableOp7Adam/batch_normalization_11/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_11/beta/m/Read/ReadVariableOp+Adam/conv3d_12/kernel/m/Read/ReadVariableOp)Adam/conv3d_12/bias/m/Read/ReadVariableOp7Adam/batch_normalization_12/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_12/beta/m/Read/ReadVariableOp+Adam/conv3d_13/kernel/m/Read/ReadVariableOp)Adam/conv3d_13/bias/m/Read/ReadVariableOp7Adam/batch_normalization_13/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_13/beta/m/Read/ReadVariableOp+Adam/conv3d_14/kernel/m/Read/ReadVariableOp)Adam/conv3d_14/bias/m/Read/ReadVariableOp+Adam/1_dense18/kernel/m/Read/ReadVariableOp)Adam/1_dense18/bias/m/Read/ReadVariableOp+Adam/2_dense32/kernel/m/Read/ReadVariableOp)Adam/2_dense32/bias/m/Read/ReadVariableOp7Adam/batch_normalization_14/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_14/beta/m/Read/ReadVariableOp+Adam/3_dense64/kernel/m/Read/ReadVariableOp)Adam/3_dense64/bias/m/Read/ReadVariableOp+Adam/4_dense32/kernel/m/Read/ReadVariableOp)Adam/4_dense32/bias/m/Read/ReadVariableOp)Adam/dense_2/kernel/m/Read/ReadVariableOp'Adam/dense_2/bias/m/Read/ReadVariableOp+Adam/5_dense18/kernel/m/Read/ReadVariableOp)Adam/5_dense18/bias/m/Read/ReadVariableOp0Adam/combined_input/kernel/m/Read/ReadVariableOp.Adam/combined_input/bias/m/Read/ReadVariableOp1Adam/ensemble_output/kernel/m/Read/ReadVariableOp/Adam/ensemble_output/bias/m/Read/ReadVariableOp+Adam/conv3d_10/kernel/v/Read/ReadVariableOp)Adam/conv3d_10/bias/v/Read/ReadVariableOp7Adam/batch_normalization_10/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_10/beta/v/Read/ReadVariableOp+Adam/conv3d_11/kernel/v/Read/ReadVariableOp)Adam/conv3d_11/bias/v/Read/ReadVariableOp7Adam/batch_normalization_11/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_11/beta/v/Read/ReadVariableOp+Adam/conv3d_12/kernel/v/Read/ReadVariableOp)Adam/conv3d_12/bias/v/Read/ReadVariableOp7Adam/batch_normalization_12/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_12/beta/v/Read/ReadVariableOp+Adam/conv3d_13/kernel/v/Read/ReadVariableOp)Adam/conv3d_13/bias/v/Read/ReadVariableOp7Adam/batch_normalization_13/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_13/beta/v/Read/ReadVariableOp+Adam/conv3d_14/kernel/v/Read/ReadVariableOp)Adam/conv3d_14/bias/v/Read/ReadVariableOp+Adam/1_dense18/kernel/v/Read/ReadVariableOp)Adam/1_dense18/bias/v/Read/ReadVariableOp+Adam/2_dense32/kernel/v/Read/ReadVariableOp)Adam/2_dense32/bias/v/Read/ReadVariableOp7Adam/batch_normalization_14/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_14/beta/v/Read/ReadVariableOp+Adam/3_dense64/kernel/v/Read/ReadVariableOp)Adam/3_dense64/bias/v/Read/ReadVariableOp+Adam/4_dense32/kernel/v/Read/ReadVariableOp)Adam/4_dense32/bias/v/Read/ReadVariableOp)Adam/dense_2/kernel/v/Read/ReadVariableOp'Adam/dense_2/bias/v/Read/ReadVariableOp+Adam/5_dense18/kernel/v/Read/ReadVariableOp)Adam/5_dense18/bias/v/Read/ReadVariableOp0Adam/combined_input/kernel/v/Read/ReadVariableOp.Adam/combined_input/bias/v/Read/ReadVariableOp1Adam/ensemble_output/kernel/v/Read/ReadVariableOp/Adam/ensemble_output/bias/v/Read/ReadVariableOpConst*?
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
 __inference__traced_save_9337779
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv3d_10/kernelconv3d_10/biasbatch_normalization_10/gammabatch_normalization_10/beta"batch_normalization_10/moving_mean&batch_normalization_10/moving_varianceconv3d_11/kernelconv3d_11/biasbatch_normalization_11/gammabatch_normalization_11/beta"batch_normalization_11/moving_mean&batch_normalization_11/moving_varianceconv3d_12/kernelconv3d_12/biasbatch_normalization_12/gammabatch_normalization_12/beta"batch_normalization_12/moving_mean&batch_normalization_12/moving_varianceconv3d_13/kernelconv3d_13/biasbatch_normalization_13/gammabatch_normalization_13/beta"batch_normalization_13/moving_mean&batch_normalization_13/moving_varianceconv3d_14/kernelconv3d_14/bias1_dense18/kernel1_dense18/bias2_dense32/kernel2_dense32/biasbatch_normalization_14/gammabatch_normalization_14/beta"batch_normalization_14/moving_mean&batch_normalization_14/moving_variance3_dense64/kernel3_dense64/bias4_dense32/kernel4_dense32/biasdense_2/kerneldense_2/bias5_dense18/kernel5_dense18/biascombined_input/kernelcombined_input/biasensemble_output/kernelensemble_output/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decaytotalcounttotal_1count_1Adam/conv3d_10/kernel/mAdam/conv3d_10/bias/m#Adam/batch_normalization_10/gamma/m"Adam/batch_normalization_10/beta/mAdam/conv3d_11/kernel/mAdam/conv3d_11/bias/m#Adam/batch_normalization_11/gamma/m"Adam/batch_normalization_11/beta/mAdam/conv3d_12/kernel/mAdam/conv3d_12/bias/m#Adam/batch_normalization_12/gamma/m"Adam/batch_normalization_12/beta/mAdam/conv3d_13/kernel/mAdam/conv3d_13/bias/m#Adam/batch_normalization_13/gamma/m"Adam/batch_normalization_13/beta/mAdam/conv3d_14/kernel/mAdam/conv3d_14/bias/mAdam/1_dense18/kernel/mAdam/1_dense18/bias/mAdam/2_dense32/kernel/mAdam/2_dense32/bias/m#Adam/batch_normalization_14/gamma/m"Adam/batch_normalization_14/beta/mAdam/3_dense64/kernel/mAdam/3_dense64/bias/mAdam/4_dense32/kernel/mAdam/4_dense32/bias/mAdam/dense_2/kernel/mAdam/dense_2/bias/mAdam/5_dense18/kernel/mAdam/5_dense18/bias/mAdam/combined_input/kernel/mAdam/combined_input/bias/mAdam/ensemble_output/kernel/mAdam/ensemble_output/bias/mAdam/conv3d_10/kernel/vAdam/conv3d_10/bias/v#Adam/batch_normalization_10/gamma/v"Adam/batch_normalization_10/beta/vAdam/conv3d_11/kernel/vAdam/conv3d_11/bias/v#Adam/batch_normalization_11/gamma/v"Adam/batch_normalization_11/beta/vAdam/conv3d_12/kernel/vAdam/conv3d_12/bias/v#Adam/batch_normalization_12/gamma/v"Adam/batch_normalization_12/beta/vAdam/conv3d_13/kernel/vAdam/conv3d_13/bias/v#Adam/batch_normalization_13/gamma/v"Adam/batch_normalization_13/beta/vAdam/conv3d_14/kernel/vAdam/conv3d_14/bias/vAdam/1_dense18/kernel/vAdam/1_dense18/bias/vAdam/2_dense32/kernel/vAdam/2_dense32/bias/v#Adam/batch_normalization_14/gamma/v"Adam/batch_normalization_14/beta/vAdam/3_dense64/kernel/vAdam/3_dense64/bias/vAdam/4_dense32/kernel/vAdam/4_dense32/bias/vAdam/dense_2/kernel/vAdam/dense_2/bias/vAdam/5_dense18/kernel/vAdam/5_dense18/bias/vAdam/combined_input/kernel/vAdam/combined_input/bias/vAdam/ensemble_output/kernel/vAdam/ensemble_output/bias/v*?
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
#__inference__traced_restore_9338167??
?
?
F__inference_1_dense18_layer_call_and_return_conditional_losses_9334332

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
8__inference_batch_normalization_13_layer_call_fn_9336919

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
GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_93348102
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
?
?
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_9336791

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
i
M__inference_max_pooling3d_12_layer_call_and_return_conditional_losses_9333630

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
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_9333857

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
?
?
+__inference_4_dense32_layer_call_fn_9337246

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
F__inference_4_dense32_layer_call_and_return_conditional_losses_93344342
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
?
?
+__inference_3_dense64_layer_call_fn_9337204

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
F__inference_3_dense64_layer_call_and_return_conditional_losses_93344002
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
?
N
2__inference_max_pooling3d_12_layer_call_fn_9336693

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
GPU2*0J 8? *V
fQRO
M__inference_max_pooling3d_12_layer_call_and_return_conditional_losses_93342192
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
?
?
1__inference_ensemble_output_layer_call_fn_9337366

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
L__inference_ensemble_output_layer_call_and_return_conditional_losses_93345012
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
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_9336955

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
?
?
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_9334869

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
?
G
+__inference_dropout_2_layer_call_fn_9337302

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
F__inference_dropout_2_layer_call_and_return_conditional_losses_93344622
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
?

?
8__inference_batch_normalization_14_layer_call_fn_9337084

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
GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_14_layer_call_and_return_conditional_losses_93339612
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
?
?
F__inference_conv3d_12_layer_call_and_return_conditional_losses_9334209

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
?
?
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_9336827

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
?
?
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_9334987

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
?
F__inference_1_dense18_layer_call_and_return_conditional_losses_9337031

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
?
?
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_9336481

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
?
?
+__inference_conv3d_11_layer_call_fn_9336508

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
GPU2*0J 8? *O
fJRH
F__inference_conv3d_11_layer_call_and_return_conditional_losses_93341592
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
?
?
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_9336645

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
?
N
2__inference_max_pooling3d_12_layer_call_fn_9336688

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
GPU2*0J 8? *V
fQRO
M__inference_max_pooling3d_12_layer_call_and_return_conditional_losses_93336302
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
8__inference_batch_normalization_12_layer_call_fn_9336716

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
GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_93336652
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
??
?8
 __inference__traced_save_9337779
file_prefix/
+savev2_conv3d_10_kernel_read_readvariableop-
)savev2_conv3d_10_bias_read_readvariableop;
7savev2_batch_normalization_10_gamma_read_readvariableop:
6savev2_batch_normalization_10_beta_read_readvariableopA
=savev2_batch_normalization_10_moving_mean_read_readvariableopE
Asavev2_batch_normalization_10_moving_variance_read_readvariableop/
+savev2_conv3d_11_kernel_read_readvariableop-
)savev2_conv3d_11_bias_read_readvariableop;
7savev2_batch_normalization_11_gamma_read_readvariableop:
6savev2_batch_normalization_11_beta_read_readvariableopA
=savev2_batch_normalization_11_moving_mean_read_readvariableopE
Asavev2_batch_normalization_11_moving_variance_read_readvariableop/
+savev2_conv3d_12_kernel_read_readvariableop-
)savev2_conv3d_12_bias_read_readvariableop;
7savev2_batch_normalization_12_gamma_read_readvariableop:
6savev2_batch_normalization_12_beta_read_readvariableopA
=savev2_batch_normalization_12_moving_mean_read_readvariableopE
Asavev2_batch_normalization_12_moving_variance_read_readvariableop/
+savev2_conv3d_13_kernel_read_readvariableop-
)savev2_conv3d_13_bias_read_readvariableop;
7savev2_batch_normalization_13_gamma_read_readvariableop:
6savev2_batch_normalization_13_beta_read_readvariableopA
=savev2_batch_normalization_13_moving_mean_read_readvariableopE
Asavev2_batch_normalization_13_moving_variance_read_readvariableop/
+savev2_conv3d_14_kernel_read_readvariableop-
)savev2_conv3d_14_bias_read_readvariableop/
+savev2_1_dense18_kernel_read_readvariableop-
)savev2_1_dense18_bias_read_readvariableop/
+savev2_2_dense32_kernel_read_readvariableop-
)savev2_2_dense32_bias_read_readvariableop;
7savev2_batch_normalization_14_gamma_read_readvariableop:
6savev2_batch_normalization_14_beta_read_readvariableopA
=savev2_batch_normalization_14_moving_mean_read_readvariableopE
Asavev2_batch_normalization_14_moving_variance_read_readvariableop/
+savev2_3_dense64_kernel_read_readvariableop-
)savev2_3_dense64_bias_read_readvariableop/
+savev2_4_dense32_kernel_read_readvariableop-
)savev2_4_dense32_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop/
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
"savev2_count_1_read_readvariableop6
2savev2_adam_conv3d_10_kernel_m_read_readvariableop4
0savev2_adam_conv3d_10_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_10_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_10_beta_m_read_readvariableop6
2savev2_adam_conv3d_11_kernel_m_read_readvariableop4
0savev2_adam_conv3d_11_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_11_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_11_beta_m_read_readvariableop6
2savev2_adam_conv3d_12_kernel_m_read_readvariableop4
0savev2_adam_conv3d_12_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_12_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_12_beta_m_read_readvariableop6
2savev2_adam_conv3d_13_kernel_m_read_readvariableop4
0savev2_adam_conv3d_13_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_13_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_13_beta_m_read_readvariableop6
2savev2_adam_conv3d_14_kernel_m_read_readvariableop4
0savev2_adam_conv3d_14_bias_m_read_readvariableop6
2savev2_adam_1_dense18_kernel_m_read_readvariableop4
0savev2_adam_1_dense18_bias_m_read_readvariableop6
2savev2_adam_2_dense32_kernel_m_read_readvariableop4
0savev2_adam_2_dense32_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_14_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_14_beta_m_read_readvariableop6
2savev2_adam_3_dense64_kernel_m_read_readvariableop4
0savev2_adam_3_dense64_bias_m_read_readvariableop6
2savev2_adam_4_dense32_kernel_m_read_readvariableop4
0savev2_adam_4_dense32_bias_m_read_readvariableop4
0savev2_adam_dense_2_kernel_m_read_readvariableop2
.savev2_adam_dense_2_bias_m_read_readvariableop6
2savev2_adam_5_dense18_kernel_m_read_readvariableop4
0savev2_adam_5_dense18_bias_m_read_readvariableop;
7savev2_adam_combined_input_kernel_m_read_readvariableop9
5savev2_adam_combined_input_bias_m_read_readvariableop<
8savev2_adam_ensemble_output_kernel_m_read_readvariableop:
6savev2_adam_ensemble_output_bias_m_read_readvariableop6
2savev2_adam_conv3d_10_kernel_v_read_readvariableop4
0savev2_adam_conv3d_10_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_10_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_10_beta_v_read_readvariableop6
2savev2_adam_conv3d_11_kernel_v_read_readvariableop4
0savev2_adam_conv3d_11_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_11_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_11_beta_v_read_readvariableop6
2savev2_adam_conv3d_12_kernel_v_read_readvariableop4
0savev2_adam_conv3d_12_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_12_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_12_beta_v_read_readvariableop6
2savev2_adam_conv3d_13_kernel_v_read_readvariableop4
0savev2_adam_conv3d_13_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_13_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_13_beta_v_read_readvariableop6
2savev2_adam_conv3d_14_kernel_v_read_readvariableop4
0savev2_adam_conv3d_14_bias_v_read_readvariableop6
2savev2_adam_1_dense18_kernel_v_read_readvariableop4
0savev2_adam_1_dense18_bias_v_read_readvariableop6
2savev2_adam_2_dense32_kernel_v_read_readvariableop4
0savev2_adam_2_dense32_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_14_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_14_beta_v_read_readvariableop6
2savev2_adam_3_dense64_kernel_v_read_readvariableop4
0savev2_adam_3_dense64_bias_v_read_readvariableop6
2savev2_adam_4_dense32_kernel_v_read_readvariableop4
0savev2_adam_4_dense32_bias_v_read_readvariableop4
0savev2_adam_dense_2_kernel_v_read_readvariableop2
.savev2_adam_dense_2_bias_v_read_readvariableop6
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
SaveV2/shape_and_slices?6
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv3d_10_kernel_read_readvariableop)savev2_conv3d_10_bias_read_readvariableop7savev2_batch_normalization_10_gamma_read_readvariableop6savev2_batch_normalization_10_beta_read_readvariableop=savev2_batch_normalization_10_moving_mean_read_readvariableopAsavev2_batch_normalization_10_moving_variance_read_readvariableop+savev2_conv3d_11_kernel_read_readvariableop)savev2_conv3d_11_bias_read_readvariableop7savev2_batch_normalization_11_gamma_read_readvariableop6savev2_batch_normalization_11_beta_read_readvariableop=savev2_batch_normalization_11_moving_mean_read_readvariableopAsavev2_batch_normalization_11_moving_variance_read_readvariableop+savev2_conv3d_12_kernel_read_readvariableop)savev2_conv3d_12_bias_read_readvariableop7savev2_batch_normalization_12_gamma_read_readvariableop6savev2_batch_normalization_12_beta_read_readvariableop=savev2_batch_normalization_12_moving_mean_read_readvariableopAsavev2_batch_normalization_12_moving_variance_read_readvariableop+savev2_conv3d_13_kernel_read_readvariableop)savev2_conv3d_13_bias_read_readvariableop7savev2_batch_normalization_13_gamma_read_readvariableop6savev2_batch_normalization_13_beta_read_readvariableop=savev2_batch_normalization_13_moving_mean_read_readvariableopAsavev2_batch_normalization_13_moving_variance_read_readvariableop+savev2_conv3d_14_kernel_read_readvariableop)savev2_conv3d_14_bias_read_readvariableop+savev2_1_dense18_kernel_read_readvariableop)savev2_1_dense18_bias_read_readvariableop+savev2_2_dense32_kernel_read_readvariableop)savev2_2_dense32_bias_read_readvariableop7savev2_batch_normalization_14_gamma_read_readvariableop6savev2_batch_normalization_14_beta_read_readvariableop=savev2_batch_normalization_14_moving_mean_read_readvariableopAsavev2_batch_normalization_14_moving_variance_read_readvariableop+savev2_3_dense64_kernel_read_readvariableop)savev2_3_dense64_bias_read_readvariableop+savev2_4_dense32_kernel_read_readvariableop)savev2_4_dense32_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop+savev2_5_dense18_kernel_read_readvariableop)savev2_5_dense18_bias_read_readvariableop0savev2_combined_input_kernel_read_readvariableop.savev2_combined_input_bias_read_readvariableop1savev2_ensemble_output_kernel_read_readvariableop/savev2_ensemble_output_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop2savev2_adam_conv3d_10_kernel_m_read_readvariableop0savev2_adam_conv3d_10_bias_m_read_readvariableop>savev2_adam_batch_normalization_10_gamma_m_read_readvariableop=savev2_adam_batch_normalization_10_beta_m_read_readvariableop2savev2_adam_conv3d_11_kernel_m_read_readvariableop0savev2_adam_conv3d_11_bias_m_read_readvariableop>savev2_adam_batch_normalization_11_gamma_m_read_readvariableop=savev2_adam_batch_normalization_11_beta_m_read_readvariableop2savev2_adam_conv3d_12_kernel_m_read_readvariableop0savev2_adam_conv3d_12_bias_m_read_readvariableop>savev2_adam_batch_normalization_12_gamma_m_read_readvariableop=savev2_adam_batch_normalization_12_beta_m_read_readvariableop2savev2_adam_conv3d_13_kernel_m_read_readvariableop0savev2_adam_conv3d_13_bias_m_read_readvariableop>savev2_adam_batch_normalization_13_gamma_m_read_readvariableop=savev2_adam_batch_normalization_13_beta_m_read_readvariableop2savev2_adam_conv3d_14_kernel_m_read_readvariableop0savev2_adam_conv3d_14_bias_m_read_readvariableop2savev2_adam_1_dense18_kernel_m_read_readvariableop0savev2_adam_1_dense18_bias_m_read_readvariableop2savev2_adam_2_dense32_kernel_m_read_readvariableop0savev2_adam_2_dense32_bias_m_read_readvariableop>savev2_adam_batch_normalization_14_gamma_m_read_readvariableop=savev2_adam_batch_normalization_14_beta_m_read_readvariableop2savev2_adam_3_dense64_kernel_m_read_readvariableop0savev2_adam_3_dense64_bias_m_read_readvariableop2savev2_adam_4_dense32_kernel_m_read_readvariableop0savev2_adam_4_dense32_bias_m_read_readvariableop0savev2_adam_dense_2_kernel_m_read_readvariableop.savev2_adam_dense_2_bias_m_read_readvariableop2savev2_adam_5_dense18_kernel_m_read_readvariableop0savev2_adam_5_dense18_bias_m_read_readvariableop7savev2_adam_combined_input_kernel_m_read_readvariableop5savev2_adam_combined_input_bias_m_read_readvariableop8savev2_adam_ensemble_output_kernel_m_read_readvariableop6savev2_adam_ensemble_output_bias_m_read_readvariableop2savev2_adam_conv3d_10_kernel_v_read_readvariableop0savev2_adam_conv3d_10_bias_v_read_readvariableop>savev2_adam_batch_normalization_10_gamma_v_read_readvariableop=savev2_adam_batch_normalization_10_beta_v_read_readvariableop2savev2_adam_conv3d_11_kernel_v_read_readvariableop0savev2_adam_conv3d_11_bias_v_read_readvariableop>savev2_adam_batch_normalization_11_gamma_v_read_readvariableop=savev2_adam_batch_normalization_11_beta_v_read_readvariableop2savev2_adam_conv3d_12_kernel_v_read_readvariableop0savev2_adam_conv3d_12_bias_v_read_readvariableop>savev2_adam_batch_normalization_12_gamma_v_read_readvariableop=savev2_adam_batch_normalization_12_beta_v_read_readvariableop2savev2_adam_conv3d_13_kernel_v_read_readvariableop0savev2_adam_conv3d_13_bias_v_read_readvariableop>savev2_adam_batch_normalization_13_gamma_v_read_readvariableop=savev2_adam_batch_normalization_13_beta_v_read_readvariableop2savev2_adam_conv3d_14_kernel_v_read_readvariableop0savev2_adam_conv3d_14_bias_v_read_readvariableop2savev2_adam_1_dense18_kernel_v_read_readvariableop0savev2_adam_1_dense18_bias_v_read_readvariableop2savev2_adam_2_dense32_kernel_v_read_readvariableop0savev2_adam_2_dense32_bias_v_read_readvariableop>savev2_adam_batch_normalization_14_gamma_v_read_readvariableop=savev2_adam_batch_normalization_14_beta_v_read_readvariableop2savev2_adam_3_dense64_kernel_v_read_readvariableop0savev2_adam_3_dense64_bias_v_read_readvariableop2savev2_adam_4_dense32_kernel_v_read_readvariableop0savev2_adam_4_dense32_bias_v_read_readvariableop0savev2_adam_dense_2_kernel_v_read_readvariableop.savev2_adam_dense_2_bias_v_read_readvariableop2savev2_adam_5_dense18_kernel_v_read_readvariableop0savev2_adam_5_dense18_bias_v_read_readvariableop7savev2_adam_combined_input_kernel_v_read_readvariableop5savev2_adam_combined_input_bias_v_read_readvariableop8savev2_adam_ensemble_output_kernel_v_read_readvariableop6savev2_adam_ensemble_output_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
?
?
F__inference_conv3d_10_layer_call_and_return_conditional_losses_9334109

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
?
?
+__inference_conv3d_10_layer_call_fn_9336344

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
GPU2*0J 8? *O
fJRH
F__inference_conv3d_10_layer_call_and_return_conditional_losses_93341092
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
?
i
M__inference_max_pooling3d_13_layer_call_and_return_conditional_losses_9336867

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
?
?
F__inference_3_dense64_layer_call_and_return_conditional_losses_9337215

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
?
?
0__inference_combined_input_layer_call_fn_9337346

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
K__inference_combined_input_layer_call_and_return_conditional_losses_93344842
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
?
N
2__inference_max_pooling3d_14_layer_call_fn_9337036

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
GPU2*0J 8? *V
fQRO
M__inference_max_pooling3d_14_layer_call_and_return_conditional_losses_93339262
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
?
d
F__inference_dropout_2_layer_call_and_return_conditional_losses_9334462

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
?
?
F__inference_conv3d_13_layer_call_and_return_conditional_losses_9334259

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
??
?'
G__inference_ensemble4d_layer_call_and_return_conditional_losses_9336152
inputs_0
inputs_1F
(conv3d_10_conv3d_readvariableop_resource: 7
)conv3d_10_biasadd_readvariableop_resource: <
.batch_normalization_10_readvariableop_resource: >
0batch_normalization_10_readvariableop_1_resource: M
?batch_normalization_10_fusedbatchnormv3_readvariableop_resource: O
Abatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource: F
(conv3d_11_conv3d_readvariableop_resource:  7
)conv3d_11_biasadd_readvariableop_resource: <
.batch_normalization_11_readvariableop_resource: >
0batch_normalization_11_readvariableop_1_resource: M
?batch_normalization_11_fusedbatchnormv3_readvariableop_resource: O
Abatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource: F
(conv3d_12_conv3d_readvariableop_resource: @7
)conv3d_12_biasadd_readvariableop_resource:@<
.batch_normalization_12_readvariableop_resource:@>
0batch_normalization_12_readvariableop_1_resource:@M
?batch_normalization_12_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_12_fusedbatchnormv3_readvariableop_1_resource:@G
(conv3d_13_conv3d_readvariableop_resource:@?8
)conv3d_13_biasadd_readvariableop_resource:	?=
.batch_normalization_13_readvariableop_resource:	??
0batch_normalization_13_readvariableop_1_resource:	?N
?batch_normalization_13_fusedbatchnormv3_readvariableop_resource:	?P
Abatch_normalization_13_fusedbatchnormv3_readvariableop_1_resource:	?H
(conv3d_14_conv3d_readvariableop_resource:??8
)conv3d_14_biasadd_readvariableop_resource:	?8
&dense18_matmul_readvariableop_resource:
5
'dense18_biasadd_readvariableop_resource:=
.batch_normalization_14_readvariableop_resource:	??
0batch_normalization_14_readvariableop_1_resource:	?N
?batch_normalization_14_fusedbatchnormv3_readvariableop_resource:	?P
Abatch_normalization_14_fusedbatchnormv3_readvariableop_1_resource:	?8
&dense32_matmul_readvariableop_resource: 5
'dense32_biasadd_readvariableop_resource: 8
&dense64_matmul_readvariableop_resource: @5
'dense64_biasadd_readvariableop_resource:@:
&dense_2_matmul_readvariableop_resource:
??6
'dense_2_biasadd_readvariableop_resource:	?:
(dense32_matmul_readvariableop_resource_0:@ 7
)dense32_biasadd_readvariableop_resource_0: :
(dense18_matmul_readvariableop_resource_0: 7
)dense18_biasadd_readvariableop_resource_0:@
-combined_input_matmul_readvariableop_resource:	?	<
.combined_input_biasadd_readvariableop_resource:	@
.ensemble_output_matmul_readvariableop_resource:	=
/ensemble_output_biasadd_readvariableop_resource:
identity?? 1_dense18/BiasAdd/ReadVariableOp?1_dense18/MatMul/ReadVariableOp? 2_dense32/BiasAdd/ReadVariableOp?2_dense32/MatMul/ReadVariableOp? 3_dense64/BiasAdd/ReadVariableOp?3_dense64/MatMul/ReadVariableOp? 4_dense32/BiasAdd/ReadVariableOp?4_dense32/MatMul/ReadVariableOp? 5_dense18/BiasAdd/ReadVariableOp?5_dense18/MatMul/ReadVariableOp?6batch_normalization_10/FusedBatchNormV3/ReadVariableOp?8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_10/ReadVariableOp?'batch_normalization_10/ReadVariableOp_1?6batch_normalization_11/FusedBatchNormV3/ReadVariableOp?8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_11/ReadVariableOp?'batch_normalization_11/ReadVariableOp_1?6batch_normalization_12/FusedBatchNormV3/ReadVariableOp?8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_12/ReadVariableOp?'batch_normalization_12/ReadVariableOp_1?6batch_normalization_13/FusedBatchNormV3/ReadVariableOp?8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_13/ReadVariableOp?'batch_normalization_13/ReadVariableOp_1?6batch_normalization_14/FusedBatchNormV3/ReadVariableOp?8batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_14/ReadVariableOp?'batch_normalization_14/ReadVariableOp_1?%combined_input/BiasAdd/ReadVariableOp?$combined_input/MatMul/ReadVariableOp? conv3d_10/BiasAdd/ReadVariableOp?conv3d_10/Conv3D/ReadVariableOp? conv3d_11/BiasAdd/ReadVariableOp?conv3d_11/Conv3D/ReadVariableOp? conv3d_12/BiasAdd/ReadVariableOp?conv3d_12/Conv3D/ReadVariableOp? conv3d_13/BiasAdd/ReadVariableOp?conv3d_13/Conv3D/ReadVariableOp? conv3d_14/BiasAdd/ReadVariableOp?conv3d_14/Conv3D/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?&ensemble_output/BiasAdd/ReadVariableOp?%ensemble_output/MatMul/ReadVariableOp?
conv3d_10/Conv3D/ReadVariableOpReadVariableOp(conv3d_10_conv3d_readvariableop_resource**
_output_shapes
: *
dtype02!
conv3d_10/Conv3D/ReadVariableOp?
conv3d_10/Conv3DConv3Dinputs_1'conv3d_10/Conv3D/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:??????????? *
paddingSAME*
strides	
2
conv3d_10/Conv3D?
 conv3d_10/BiasAdd/ReadVariableOpReadVariableOp)conv3d_10_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv3d_10/BiasAdd/ReadVariableOp?
conv3d_10/BiasAddBiasAddconv3d_10/Conv3D:output:0(conv3d_10/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:??????????? 2
conv3d_10/BiasAdd?
conv3d_10/LeakyRelu	LeakyReluconv3d_10/BiasAdd:output:0*5
_output_shapes#
!:??????????? 2
conv3d_10/LeakyRelu?
max_pooling3d_10/MaxPool3D	MaxPool3D!conv3d_10/LeakyRelu:activations:0*
T0*3
_output_shapes!
:?????????@@ *
ksize	
*
paddingVALID*
strides	
2
max_pooling3d_10/MaxPool3D?
%batch_normalization_10/ReadVariableOpReadVariableOp.batch_normalization_10_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_10/ReadVariableOp?
'batch_normalization_10/ReadVariableOp_1ReadVariableOp0batch_normalization_10_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_10/ReadVariableOp_1?
6batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_10/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_10/FusedBatchNormV3FusedBatchNormV3#max_pooling3d_10/MaxPool3D:output:0-batch_normalization_10/ReadVariableOp:value:0/batch_normalization_10/ReadVariableOp_1:value:0>batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*O
_output_shapes=
;:?????????@@ : : : : :*
data_formatNDHWC*
epsilon%o?:*
is_training( 2)
'batch_normalization_10/FusedBatchNormV3?
conv3d_11/Conv3D/ReadVariableOpReadVariableOp(conv3d_11_conv3d_readvariableop_resource**
_output_shapes
:  *
dtype02!
conv3d_11/Conv3D/ReadVariableOp?
conv3d_11/Conv3DConv3D+batch_normalization_10/FusedBatchNormV3:y:0'conv3d_11/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@@ *
paddingSAME*
strides	
2
conv3d_11/Conv3D?
 conv3d_11/BiasAdd/ReadVariableOpReadVariableOp)conv3d_11_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv3d_11/BiasAdd/ReadVariableOp?
conv3d_11/BiasAddBiasAddconv3d_11/Conv3D:output:0(conv3d_11/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@@ 2
conv3d_11/BiasAdd?
conv3d_11/LeakyRelu	LeakyReluconv3d_11/BiasAdd:output:0*3
_output_shapes!
:?????????@@ 2
conv3d_11/LeakyRelu?
max_pooling3d_11/MaxPool3D	MaxPool3D!conv3d_11/LeakyRelu:activations:0*
T0*3
_output_shapes!
:?????????@@ *
ksize	
*
paddingVALID*
strides	
2
max_pooling3d_11/MaxPool3D?
%batch_normalization_11/ReadVariableOpReadVariableOp.batch_normalization_11_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_11/ReadVariableOp?
'batch_normalization_11/ReadVariableOp_1ReadVariableOp0batch_normalization_11_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_11/ReadVariableOp_1?
6batch_normalization_11/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_11_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_11/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_11/FusedBatchNormV3FusedBatchNormV3#max_pooling3d_11/MaxPool3D:output:0-batch_normalization_11/ReadVariableOp:value:0/batch_normalization_11/ReadVariableOp_1:value:0>batch_normalization_11/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*O
_output_shapes=
;:?????????@@ : : : : :*
data_formatNDHWC*
epsilon%o?:*
is_training( 2)
'batch_normalization_11/FusedBatchNormV3?
conv3d_12/Conv3D/ReadVariableOpReadVariableOp(conv3d_12_conv3d_readvariableop_resource**
_output_shapes
: @*
dtype02!
conv3d_12/Conv3D/ReadVariableOp?
conv3d_12/Conv3DConv3D+batch_normalization_11/FusedBatchNormV3:y:0'conv3d_12/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@@@*
paddingSAME*
strides	
2
conv3d_12/Conv3D?
 conv3d_12/BiasAdd/ReadVariableOpReadVariableOp)conv3d_12_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv3d_12/BiasAdd/ReadVariableOp?
conv3d_12/BiasAddBiasAddconv3d_12/Conv3D:output:0(conv3d_12/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@@@2
conv3d_12/BiasAdd?
conv3d_12/LeakyRelu	LeakyReluconv3d_12/BiasAdd:output:0*3
_output_shapes!
:?????????@@@2
conv3d_12/LeakyRelu?
max_pooling3d_12/MaxPool3D	MaxPool3D!conv3d_12/LeakyRelu:activations:0*
T0*3
_output_shapes!
:?????????@@@*
ksize	
*
paddingVALID*
strides	
2
max_pooling3d_12/MaxPool3D?
%batch_normalization_12/ReadVariableOpReadVariableOp.batch_normalization_12_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_12/ReadVariableOp?
'batch_normalization_12/ReadVariableOp_1ReadVariableOp0batch_normalization_12_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_12/ReadVariableOp_1?
6batch_normalization_12/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_12_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_12/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_12_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_12/FusedBatchNormV3FusedBatchNormV3#max_pooling3d_12/MaxPool3D:output:0-batch_normalization_12/ReadVariableOp:value:0/batch_normalization_12/ReadVariableOp_1:value:0>batch_normalization_12/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*O
_output_shapes=
;:?????????@@@:@:@:@:@:*
data_formatNDHWC*
epsilon%o?:*
is_training( 2)
'batch_normalization_12/FusedBatchNormV3?
conv3d_13/Conv3D/ReadVariableOpReadVariableOp(conv3d_13_conv3d_readvariableop_resource*+
_output_shapes
:@?*
dtype02!
conv3d_13/Conv3D/ReadVariableOp?
conv3d_13/Conv3DConv3D+batch_normalization_12/FusedBatchNormV3:y:0'conv3d_13/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :?????????@@?*
paddingSAME*
strides	
2
conv3d_13/Conv3D?
 conv3d_13/BiasAdd/ReadVariableOpReadVariableOp)conv3d_13_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv3d_13/BiasAdd/ReadVariableOp?
conv3d_13/BiasAddBiasAddconv3d_13/Conv3D:output:0(conv3d_13/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :?????????@@?2
conv3d_13/BiasAdd?
conv3d_13/LeakyRelu	LeakyReluconv3d_13/BiasAdd:output:0*4
_output_shapes"
 :?????????@@?2
conv3d_13/LeakyRelu?
max_pooling3d_13/MaxPool3D	MaxPool3D!conv3d_13/LeakyRelu:activations:0*
T0*4
_output_shapes"
 :?????????@@?*
ksize	
*
paddingVALID*
strides	
2
max_pooling3d_13/MaxPool3D?
%batch_normalization_13/ReadVariableOpReadVariableOp.batch_normalization_13_readvariableop_resource*
_output_shapes	
:?*
dtype02'
%batch_normalization_13/ReadVariableOp?
'batch_normalization_13/ReadVariableOp_1ReadVariableOp0batch_normalization_13_readvariableop_1_resource*
_output_shapes	
:?*
dtype02)
'batch_normalization_13/ReadVariableOp_1?
6batch_normalization_13/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_13_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype028
6batch_normalization_13/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_13_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02:
8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_13/FusedBatchNormV3FusedBatchNormV3#max_pooling3d_13/MaxPool3D:output:0-batch_normalization_13/ReadVariableOp:value:0/batch_normalization_13/ReadVariableOp_1:value:0>batch_normalization_13/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*T
_output_shapesB
@:?????????@@?:?:?:?:?:*
data_formatNDHWC*
epsilon%o?:*
is_training( 2)
'batch_normalization_13/FusedBatchNormV3?
conv3d_14/Conv3D/ReadVariableOpReadVariableOp(conv3d_14_conv3d_readvariableop_resource*,
_output_shapes
:??*
dtype02!
conv3d_14/Conv3D/ReadVariableOp?
conv3d_14/Conv3DConv3D+batch_normalization_13/FusedBatchNormV3:y:0'conv3d_14/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :?????????@@?*
paddingSAME*
strides	
2
conv3d_14/Conv3D?
 conv3d_14/BiasAdd/ReadVariableOpReadVariableOp)conv3d_14_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv3d_14/BiasAdd/ReadVariableOp?
conv3d_14/BiasAddBiasAddconv3d_14/Conv3D:output:0(conv3d_14/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :?????????@@?2
conv3d_14/BiasAdd?
conv3d_14/LeakyRelu	LeakyReluconv3d_14/BiasAdd:output:0*4
_output_shapes"
 :?????????@@?2
conv3d_14/LeakyRelu?
max_pooling3d_14/MaxPool3D	MaxPool3D!conv3d_14/LeakyRelu:activations:0*
T0*4
_output_shapes"
 :?????????@@?*
ksize	
*
paddingVALID*
strides	
2
max_pooling3d_14/MaxPool3D?
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
%batch_normalization_14/ReadVariableOpReadVariableOp.batch_normalization_14_readvariableop_resource*
_output_shapes	
:?*
dtype02'
%batch_normalization_14/ReadVariableOp?
'batch_normalization_14/ReadVariableOp_1ReadVariableOp0batch_normalization_14_readvariableop_1_resource*
_output_shapes	
:?*
dtype02)
'batch_normalization_14/ReadVariableOp_1?
6batch_normalization_14/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_14_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype028
6batch_normalization_14/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_14_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02:
8batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_14/FusedBatchNormV3FusedBatchNormV3#max_pooling3d_14/MaxPool3D:output:0-batch_normalization_14/ReadVariableOp:value:0/batch_normalization_14/ReadVariableOp_1:value:0>batch_normalization_14/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*T
_output_shapesB
@:?????????@@?:?:?:?:?:*
data_formatNDHWC*
epsilon%o?:*
is_training( 2)
'batch_normalization_14/FusedBatchNormV3?
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
1global_average_pooling3d_2/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         23
1global_average_pooling3d_2/Mean/reduction_indices?
global_average_pooling3d_2/MeanMean+batch_normalization_14/FusedBatchNormV3:y:0:global_average_pooling3d_2/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2!
global_average_pooling3d_2/Mean?
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
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMul(global_average_pooling3d_2/Mean:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_2/BiasAddw
dense_2/LeakyRelu	LeakyReludense_2/BiasAdd:output:0*(
_output_shapes
:??????????2
dense_2/LeakyRelu?
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
dropout_2/IdentityIdentitydense_2/LeakyRelu:activations:0*
T0*(
_output_shapes
:??????????2
dropout_2/Identityx
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_1/concat/axis?
concatenate_1/concatConcatV2!5_dense18/LeakyRelu:activations:0dropout_2/Identity:output:0"concatenate_1/concat/axis:output:0*
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
NoOpNoOp!^1_dense18/BiasAdd/ReadVariableOp ^1_dense18/MatMul/ReadVariableOp!^2_dense32/BiasAdd/ReadVariableOp ^2_dense32/MatMul/ReadVariableOp!^3_dense64/BiasAdd/ReadVariableOp ^3_dense64/MatMul/ReadVariableOp!^4_dense32/BiasAdd/ReadVariableOp ^4_dense32/MatMul/ReadVariableOp!^5_dense18/BiasAdd/ReadVariableOp ^5_dense18/MatMul/ReadVariableOp7^batch_normalization_10/FusedBatchNormV3/ReadVariableOp9^batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_10/ReadVariableOp(^batch_normalization_10/ReadVariableOp_17^batch_normalization_11/FusedBatchNormV3/ReadVariableOp9^batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_11/ReadVariableOp(^batch_normalization_11/ReadVariableOp_17^batch_normalization_12/FusedBatchNormV3/ReadVariableOp9^batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_12/ReadVariableOp(^batch_normalization_12/ReadVariableOp_17^batch_normalization_13/FusedBatchNormV3/ReadVariableOp9^batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_13/ReadVariableOp(^batch_normalization_13/ReadVariableOp_17^batch_normalization_14/FusedBatchNormV3/ReadVariableOp9^batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_14/ReadVariableOp(^batch_normalization_14/ReadVariableOp_1&^combined_input/BiasAdd/ReadVariableOp%^combined_input/MatMul/ReadVariableOp!^conv3d_10/BiasAdd/ReadVariableOp ^conv3d_10/Conv3D/ReadVariableOp!^conv3d_11/BiasAdd/ReadVariableOp ^conv3d_11/Conv3D/ReadVariableOp!^conv3d_12/BiasAdd/ReadVariableOp ^conv3d_12/Conv3D/ReadVariableOp!^conv3d_13/BiasAdd/ReadVariableOp ^conv3d_13/Conv3D/ReadVariableOp!^conv3d_14/BiasAdd/ReadVariableOp ^conv3d_14/Conv3D/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp'^ensemble_output/BiasAdd/ReadVariableOp&^ensemble_output/MatMul/ReadVariableOp*"
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
5_dense18/MatMul/ReadVariableOp5_dense18/MatMul/ReadVariableOp2p
6batch_normalization_10/FusedBatchNormV3/ReadVariableOp6batch_normalization_10/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_18batch_normalization_10/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_10/ReadVariableOp%batch_normalization_10/ReadVariableOp2R
'batch_normalization_10/ReadVariableOp_1'batch_normalization_10/ReadVariableOp_12p
6batch_normalization_11/FusedBatchNormV3/ReadVariableOp6batch_normalization_11/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_18batch_normalization_11/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_11/ReadVariableOp%batch_normalization_11/ReadVariableOp2R
'batch_normalization_11/ReadVariableOp_1'batch_normalization_11/ReadVariableOp_12p
6batch_normalization_12/FusedBatchNormV3/ReadVariableOp6batch_normalization_12/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_18batch_normalization_12/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_12/ReadVariableOp%batch_normalization_12/ReadVariableOp2R
'batch_normalization_12/ReadVariableOp_1'batch_normalization_12/ReadVariableOp_12p
6batch_normalization_13/FusedBatchNormV3/ReadVariableOp6batch_normalization_13/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_18batch_normalization_13/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_13/ReadVariableOp%batch_normalization_13/ReadVariableOp2R
'batch_normalization_13/ReadVariableOp_1'batch_normalization_13/ReadVariableOp_12p
6batch_normalization_14/FusedBatchNormV3/ReadVariableOp6batch_normalization_14/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_14/FusedBatchNormV3/ReadVariableOp_18batch_normalization_14/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_14/ReadVariableOp%batch_normalization_14/ReadVariableOp2R
'batch_normalization_14/ReadVariableOp_1'batch_normalization_14/ReadVariableOp_12N
%combined_input/BiasAdd/ReadVariableOp%combined_input/BiasAdd/ReadVariableOp2L
$combined_input/MatMul/ReadVariableOp$combined_input/MatMul/ReadVariableOp2D
 conv3d_10/BiasAdd/ReadVariableOp conv3d_10/BiasAdd/ReadVariableOp2B
conv3d_10/Conv3D/ReadVariableOpconv3d_10/Conv3D/ReadVariableOp2D
 conv3d_11/BiasAdd/ReadVariableOp conv3d_11/BiasAdd/ReadVariableOp2B
conv3d_11/Conv3D/ReadVariableOpconv3d_11/Conv3D/ReadVariableOp2D
 conv3d_12/BiasAdd/ReadVariableOp conv3d_12/BiasAdd/ReadVariableOp2B
conv3d_12/Conv3D/ReadVariableOpconv3d_12/Conv3D/ReadVariableOp2D
 conv3d_13/BiasAdd/ReadVariableOp conv3d_13/BiasAdd/ReadVariableOp2B
conv3d_13/Conv3D/ReadVariableOpconv3d_13/Conv3D/ReadVariableOp2D
 conv3d_14/BiasAdd/ReadVariableOp conv3d_14/BiasAdd/ReadVariableOp2B
conv3d_14/Conv3D/ReadVariableOpconv3d_14/Conv3D/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2P
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
?
t
J__inference_concatenate_1_layer_call_and_return_conditional_losses_9334471

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
?
?
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_9334928

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
??
?*
G__inference_ensemble4d_layer_call_and_return_conditional_losses_9336335
inputs_0
inputs_1F
(conv3d_10_conv3d_readvariableop_resource: 7
)conv3d_10_biasadd_readvariableop_resource: <
.batch_normalization_10_readvariableop_resource: >
0batch_normalization_10_readvariableop_1_resource: M
?batch_normalization_10_fusedbatchnormv3_readvariableop_resource: O
Abatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource: F
(conv3d_11_conv3d_readvariableop_resource:  7
)conv3d_11_biasadd_readvariableop_resource: <
.batch_normalization_11_readvariableop_resource: >
0batch_normalization_11_readvariableop_1_resource: M
?batch_normalization_11_fusedbatchnormv3_readvariableop_resource: O
Abatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource: F
(conv3d_12_conv3d_readvariableop_resource: @7
)conv3d_12_biasadd_readvariableop_resource:@<
.batch_normalization_12_readvariableop_resource:@>
0batch_normalization_12_readvariableop_1_resource:@M
?batch_normalization_12_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_12_fusedbatchnormv3_readvariableop_1_resource:@G
(conv3d_13_conv3d_readvariableop_resource:@?8
)conv3d_13_biasadd_readvariableop_resource:	?=
.batch_normalization_13_readvariableop_resource:	??
0batch_normalization_13_readvariableop_1_resource:	?N
?batch_normalization_13_fusedbatchnormv3_readvariableop_resource:	?P
Abatch_normalization_13_fusedbatchnormv3_readvariableop_1_resource:	?H
(conv3d_14_conv3d_readvariableop_resource:??8
)conv3d_14_biasadd_readvariableop_resource:	?8
&dense18_matmul_readvariableop_resource:
5
'dense18_biasadd_readvariableop_resource:=
.batch_normalization_14_readvariableop_resource:	??
0batch_normalization_14_readvariableop_1_resource:	?N
?batch_normalization_14_fusedbatchnormv3_readvariableop_resource:	?P
Abatch_normalization_14_fusedbatchnormv3_readvariableop_1_resource:	?8
&dense32_matmul_readvariableop_resource: 5
'dense32_biasadd_readvariableop_resource: 8
&dense64_matmul_readvariableop_resource: @5
'dense64_biasadd_readvariableop_resource:@:
&dense_2_matmul_readvariableop_resource:
??6
'dense_2_biasadd_readvariableop_resource:	?:
(dense32_matmul_readvariableop_resource_0:@ 7
)dense32_biasadd_readvariableop_resource_0: :
(dense18_matmul_readvariableop_resource_0: 7
)dense18_biasadd_readvariableop_resource_0:@
-combined_input_matmul_readvariableop_resource:	?	<
.combined_input_biasadd_readvariableop_resource:	@
.ensemble_output_matmul_readvariableop_resource:	=
/ensemble_output_biasadd_readvariableop_resource:
identity?? 1_dense18/BiasAdd/ReadVariableOp?1_dense18/MatMul/ReadVariableOp? 2_dense32/BiasAdd/ReadVariableOp?2_dense32/MatMul/ReadVariableOp? 3_dense64/BiasAdd/ReadVariableOp?3_dense64/MatMul/ReadVariableOp? 4_dense32/BiasAdd/ReadVariableOp?4_dense32/MatMul/ReadVariableOp? 5_dense18/BiasAdd/ReadVariableOp?5_dense18/MatMul/ReadVariableOp?%batch_normalization_10/AssignNewValue?'batch_normalization_10/AssignNewValue_1?6batch_normalization_10/FusedBatchNormV3/ReadVariableOp?8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_10/ReadVariableOp?'batch_normalization_10/ReadVariableOp_1?%batch_normalization_11/AssignNewValue?'batch_normalization_11/AssignNewValue_1?6batch_normalization_11/FusedBatchNormV3/ReadVariableOp?8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_11/ReadVariableOp?'batch_normalization_11/ReadVariableOp_1?%batch_normalization_12/AssignNewValue?'batch_normalization_12/AssignNewValue_1?6batch_normalization_12/FusedBatchNormV3/ReadVariableOp?8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_12/ReadVariableOp?'batch_normalization_12/ReadVariableOp_1?%batch_normalization_13/AssignNewValue?'batch_normalization_13/AssignNewValue_1?6batch_normalization_13/FusedBatchNormV3/ReadVariableOp?8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_13/ReadVariableOp?'batch_normalization_13/ReadVariableOp_1?%batch_normalization_14/AssignNewValue?'batch_normalization_14/AssignNewValue_1?6batch_normalization_14/FusedBatchNormV3/ReadVariableOp?8batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_14/ReadVariableOp?'batch_normalization_14/ReadVariableOp_1?%combined_input/BiasAdd/ReadVariableOp?$combined_input/MatMul/ReadVariableOp? conv3d_10/BiasAdd/ReadVariableOp?conv3d_10/Conv3D/ReadVariableOp? conv3d_11/BiasAdd/ReadVariableOp?conv3d_11/Conv3D/ReadVariableOp? conv3d_12/BiasAdd/ReadVariableOp?conv3d_12/Conv3D/ReadVariableOp? conv3d_13/BiasAdd/ReadVariableOp?conv3d_13/Conv3D/ReadVariableOp? conv3d_14/BiasAdd/ReadVariableOp?conv3d_14/Conv3D/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?&ensemble_output/BiasAdd/ReadVariableOp?%ensemble_output/MatMul/ReadVariableOp?
conv3d_10/Conv3D/ReadVariableOpReadVariableOp(conv3d_10_conv3d_readvariableop_resource**
_output_shapes
: *
dtype02!
conv3d_10/Conv3D/ReadVariableOp?
conv3d_10/Conv3DConv3Dinputs_1'conv3d_10/Conv3D/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:??????????? *
paddingSAME*
strides	
2
conv3d_10/Conv3D?
 conv3d_10/BiasAdd/ReadVariableOpReadVariableOp)conv3d_10_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv3d_10/BiasAdd/ReadVariableOp?
conv3d_10/BiasAddBiasAddconv3d_10/Conv3D:output:0(conv3d_10/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:??????????? 2
conv3d_10/BiasAdd?
conv3d_10/LeakyRelu	LeakyReluconv3d_10/BiasAdd:output:0*5
_output_shapes#
!:??????????? 2
conv3d_10/LeakyRelu?
max_pooling3d_10/MaxPool3D	MaxPool3D!conv3d_10/LeakyRelu:activations:0*
T0*3
_output_shapes!
:?????????@@ *
ksize	
*
paddingVALID*
strides	
2
max_pooling3d_10/MaxPool3D?
%batch_normalization_10/ReadVariableOpReadVariableOp.batch_normalization_10_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_10/ReadVariableOp?
'batch_normalization_10/ReadVariableOp_1ReadVariableOp0batch_normalization_10_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_10/ReadVariableOp_1?
6batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_10/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_10/FusedBatchNormV3FusedBatchNormV3#max_pooling3d_10/MaxPool3D:output:0-batch_normalization_10/ReadVariableOp:value:0/batch_normalization_10/ReadVariableOp_1:value:0>batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*O
_output_shapes=
;:?????????@@ : : : : :*
data_formatNDHWC*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_10/FusedBatchNormV3?
%batch_normalization_10/AssignNewValueAssignVariableOp?batch_normalization_10_fusedbatchnormv3_readvariableop_resource4batch_normalization_10/FusedBatchNormV3:batch_mean:07^batch_normalization_10/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_10/AssignNewValue?
'batch_normalization_10/AssignNewValue_1AssignVariableOpAbatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_10/FusedBatchNormV3:batch_variance:09^batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02)
'batch_normalization_10/AssignNewValue_1?
conv3d_11/Conv3D/ReadVariableOpReadVariableOp(conv3d_11_conv3d_readvariableop_resource**
_output_shapes
:  *
dtype02!
conv3d_11/Conv3D/ReadVariableOp?
conv3d_11/Conv3DConv3D+batch_normalization_10/FusedBatchNormV3:y:0'conv3d_11/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@@ *
paddingSAME*
strides	
2
conv3d_11/Conv3D?
 conv3d_11/BiasAdd/ReadVariableOpReadVariableOp)conv3d_11_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv3d_11/BiasAdd/ReadVariableOp?
conv3d_11/BiasAddBiasAddconv3d_11/Conv3D:output:0(conv3d_11/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@@ 2
conv3d_11/BiasAdd?
conv3d_11/LeakyRelu	LeakyReluconv3d_11/BiasAdd:output:0*3
_output_shapes!
:?????????@@ 2
conv3d_11/LeakyRelu?
max_pooling3d_11/MaxPool3D	MaxPool3D!conv3d_11/LeakyRelu:activations:0*
T0*3
_output_shapes!
:?????????@@ *
ksize	
*
paddingVALID*
strides	
2
max_pooling3d_11/MaxPool3D?
%batch_normalization_11/ReadVariableOpReadVariableOp.batch_normalization_11_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_11/ReadVariableOp?
'batch_normalization_11/ReadVariableOp_1ReadVariableOp0batch_normalization_11_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_11/ReadVariableOp_1?
6batch_normalization_11/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_11_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_11/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_11/FusedBatchNormV3FusedBatchNormV3#max_pooling3d_11/MaxPool3D:output:0-batch_normalization_11/ReadVariableOp:value:0/batch_normalization_11/ReadVariableOp_1:value:0>batch_normalization_11/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*O
_output_shapes=
;:?????????@@ : : : : :*
data_formatNDHWC*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_11/FusedBatchNormV3?
%batch_normalization_11/AssignNewValueAssignVariableOp?batch_normalization_11_fusedbatchnormv3_readvariableop_resource4batch_normalization_11/FusedBatchNormV3:batch_mean:07^batch_normalization_11/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_11/AssignNewValue?
'batch_normalization_11/AssignNewValue_1AssignVariableOpAbatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_11/FusedBatchNormV3:batch_variance:09^batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02)
'batch_normalization_11/AssignNewValue_1?
conv3d_12/Conv3D/ReadVariableOpReadVariableOp(conv3d_12_conv3d_readvariableop_resource**
_output_shapes
: @*
dtype02!
conv3d_12/Conv3D/ReadVariableOp?
conv3d_12/Conv3DConv3D+batch_normalization_11/FusedBatchNormV3:y:0'conv3d_12/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@@@*
paddingSAME*
strides	
2
conv3d_12/Conv3D?
 conv3d_12/BiasAdd/ReadVariableOpReadVariableOp)conv3d_12_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv3d_12/BiasAdd/ReadVariableOp?
conv3d_12/BiasAddBiasAddconv3d_12/Conv3D:output:0(conv3d_12/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@@@2
conv3d_12/BiasAdd?
conv3d_12/LeakyRelu	LeakyReluconv3d_12/BiasAdd:output:0*3
_output_shapes!
:?????????@@@2
conv3d_12/LeakyRelu?
max_pooling3d_12/MaxPool3D	MaxPool3D!conv3d_12/LeakyRelu:activations:0*
T0*3
_output_shapes!
:?????????@@@*
ksize	
*
paddingVALID*
strides	
2
max_pooling3d_12/MaxPool3D?
%batch_normalization_12/ReadVariableOpReadVariableOp.batch_normalization_12_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_12/ReadVariableOp?
'batch_normalization_12/ReadVariableOp_1ReadVariableOp0batch_normalization_12_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_12/ReadVariableOp_1?
6batch_normalization_12/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_12_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_12/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_12_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_12/FusedBatchNormV3FusedBatchNormV3#max_pooling3d_12/MaxPool3D:output:0-batch_normalization_12/ReadVariableOp:value:0/batch_normalization_12/ReadVariableOp_1:value:0>batch_normalization_12/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*O
_output_shapes=
;:?????????@@@:@:@:@:@:*
data_formatNDHWC*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_12/FusedBatchNormV3?
%batch_normalization_12/AssignNewValueAssignVariableOp?batch_normalization_12_fusedbatchnormv3_readvariableop_resource4batch_normalization_12/FusedBatchNormV3:batch_mean:07^batch_normalization_12/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_12/AssignNewValue?
'batch_normalization_12/AssignNewValue_1AssignVariableOpAbatch_normalization_12_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_12/FusedBatchNormV3:batch_variance:09^batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02)
'batch_normalization_12/AssignNewValue_1?
conv3d_13/Conv3D/ReadVariableOpReadVariableOp(conv3d_13_conv3d_readvariableop_resource*+
_output_shapes
:@?*
dtype02!
conv3d_13/Conv3D/ReadVariableOp?
conv3d_13/Conv3DConv3D+batch_normalization_12/FusedBatchNormV3:y:0'conv3d_13/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :?????????@@?*
paddingSAME*
strides	
2
conv3d_13/Conv3D?
 conv3d_13/BiasAdd/ReadVariableOpReadVariableOp)conv3d_13_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv3d_13/BiasAdd/ReadVariableOp?
conv3d_13/BiasAddBiasAddconv3d_13/Conv3D:output:0(conv3d_13/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :?????????@@?2
conv3d_13/BiasAdd?
conv3d_13/LeakyRelu	LeakyReluconv3d_13/BiasAdd:output:0*4
_output_shapes"
 :?????????@@?2
conv3d_13/LeakyRelu?
max_pooling3d_13/MaxPool3D	MaxPool3D!conv3d_13/LeakyRelu:activations:0*
T0*4
_output_shapes"
 :?????????@@?*
ksize	
*
paddingVALID*
strides	
2
max_pooling3d_13/MaxPool3D?
%batch_normalization_13/ReadVariableOpReadVariableOp.batch_normalization_13_readvariableop_resource*
_output_shapes	
:?*
dtype02'
%batch_normalization_13/ReadVariableOp?
'batch_normalization_13/ReadVariableOp_1ReadVariableOp0batch_normalization_13_readvariableop_1_resource*
_output_shapes	
:?*
dtype02)
'batch_normalization_13/ReadVariableOp_1?
6batch_normalization_13/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_13_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype028
6batch_normalization_13/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_13_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02:
8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_13/FusedBatchNormV3FusedBatchNormV3#max_pooling3d_13/MaxPool3D:output:0-batch_normalization_13/ReadVariableOp:value:0/batch_normalization_13/ReadVariableOp_1:value:0>batch_normalization_13/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*T
_output_shapesB
@:?????????@@?:?:?:?:?:*
data_formatNDHWC*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_13/FusedBatchNormV3?
%batch_normalization_13/AssignNewValueAssignVariableOp?batch_normalization_13_fusedbatchnormv3_readvariableop_resource4batch_normalization_13/FusedBatchNormV3:batch_mean:07^batch_normalization_13/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_13/AssignNewValue?
'batch_normalization_13/AssignNewValue_1AssignVariableOpAbatch_normalization_13_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_13/FusedBatchNormV3:batch_variance:09^batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02)
'batch_normalization_13/AssignNewValue_1?
conv3d_14/Conv3D/ReadVariableOpReadVariableOp(conv3d_14_conv3d_readvariableop_resource*,
_output_shapes
:??*
dtype02!
conv3d_14/Conv3D/ReadVariableOp?
conv3d_14/Conv3DConv3D+batch_normalization_13/FusedBatchNormV3:y:0'conv3d_14/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :?????????@@?*
paddingSAME*
strides	
2
conv3d_14/Conv3D?
 conv3d_14/BiasAdd/ReadVariableOpReadVariableOp)conv3d_14_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv3d_14/BiasAdd/ReadVariableOp?
conv3d_14/BiasAddBiasAddconv3d_14/Conv3D:output:0(conv3d_14/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :?????????@@?2
conv3d_14/BiasAdd?
conv3d_14/LeakyRelu	LeakyReluconv3d_14/BiasAdd:output:0*4
_output_shapes"
 :?????????@@?2
conv3d_14/LeakyRelu?
max_pooling3d_14/MaxPool3D	MaxPool3D!conv3d_14/LeakyRelu:activations:0*
T0*4
_output_shapes"
 :?????????@@?*
ksize	
*
paddingVALID*
strides	
2
max_pooling3d_14/MaxPool3D?
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
%batch_normalization_14/ReadVariableOpReadVariableOp.batch_normalization_14_readvariableop_resource*
_output_shapes	
:?*
dtype02'
%batch_normalization_14/ReadVariableOp?
'batch_normalization_14/ReadVariableOp_1ReadVariableOp0batch_normalization_14_readvariableop_1_resource*
_output_shapes	
:?*
dtype02)
'batch_normalization_14/ReadVariableOp_1?
6batch_normalization_14/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_14_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype028
6batch_normalization_14/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_14_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02:
8batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_14/FusedBatchNormV3FusedBatchNormV3#max_pooling3d_14/MaxPool3D:output:0-batch_normalization_14/ReadVariableOp:value:0/batch_normalization_14/ReadVariableOp_1:value:0>batch_normalization_14/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*T
_output_shapesB
@:?????????@@?:?:?:?:?:*
data_formatNDHWC*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_14/FusedBatchNormV3?
%batch_normalization_14/AssignNewValueAssignVariableOp?batch_normalization_14_fusedbatchnormv3_readvariableop_resource4batch_normalization_14/FusedBatchNormV3:batch_mean:07^batch_normalization_14/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_14/AssignNewValue?
'batch_normalization_14/AssignNewValue_1AssignVariableOpAbatch_normalization_14_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_14/FusedBatchNormV3:batch_variance:09^batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02)
'batch_normalization_14/AssignNewValue_1?
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
1global_average_pooling3d_2/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         23
1global_average_pooling3d_2/Mean/reduction_indices?
global_average_pooling3d_2/MeanMean+batch_normalization_14/FusedBatchNormV3:y:0:global_average_pooling3d_2/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2!
global_average_pooling3d_2/Mean?
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
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMul(global_average_pooling3d_2/Mean:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_2/BiasAddw
dense_2/LeakyRelu	LeakyReludense_2/BiasAdd:output:0*(
_output_shapes
:??????????2
dense_2/LeakyRelu?
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
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout_2/dropout/Const?
dropout_2/dropout/MulMuldense_2/LeakyRelu:activations:0 dropout_2/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout_2/dropout/Mul?
dropout_2/dropout/ShapeShapedense_2/LeakyRelu:activations:0*
T0*
_output_shapes
:2
dropout_2/dropout/Shape?
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype020
.dropout_2/dropout/random_uniform/RandomUniform{
dropout_2/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *???>2
dropout_2/dropout/Const_1?
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0"dropout_2/dropout/Const_1:output:0*
T0*(
_output_shapes
:??????????2 
dropout_2/dropout/GreaterEqual?
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout_2/dropout/Cast?
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout_2/dropout/Mul_1x
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_1/concat/axis?
concatenate_1/concatConcatV2!5_dense18/LeakyRelu:activations:0dropout_2/dropout/Mul_1:z:0"concatenate_1/concat/axis:output:0*
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
NoOpNoOp!^1_dense18/BiasAdd/ReadVariableOp ^1_dense18/MatMul/ReadVariableOp!^2_dense32/BiasAdd/ReadVariableOp ^2_dense32/MatMul/ReadVariableOp!^3_dense64/BiasAdd/ReadVariableOp ^3_dense64/MatMul/ReadVariableOp!^4_dense32/BiasAdd/ReadVariableOp ^4_dense32/MatMul/ReadVariableOp!^5_dense18/BiasAdd/ReadVariableOp ^5_dense18/MatMul/ReadVariableOp&^batch_normalization_10/AssignNewValue(^batch_normalization_10/AssignNewValue_17^batch_normalization_10/FusedBatchNormV3/ReadVariableOp9^batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_10/ReadVariableOp(^batch_normalization_10/ReadVariableOp_1&^batch_normalization_11/AssignNewValue(^batch_normalization_11/AssignNewValue_17^batch_normalization_11/FusedBatchNormV3/ReadVariableOp9^batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_11/ReadVariableOp(^batch_normalization_11/ReadVariableOp_1&^batch_normalization_12/AssignNewValue(^batch_normalization_12/AssignNewValue_17^batch_normalization_12/FusedBatchNormV3/ReadVariableOp9^batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_12/ReadVariableOp(^batch_normalization_12/ReadVariableOp_1&^batch_normalization_13/AssignNewValue(^batch_normalization_13/AssignNewValue_17^batch_normalization_13/FusedBatchNormV3/ReadVariableOp9^batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_13/ReadVariableOp(^batch_normalization_13/ReadVariableOp_1&^batch_normalization_14/AssignNewValue(^batch_normalization_14/AssignNewValue_17^batch_normalization_14/FusedBatchNormV3/ReadVariableOp9^batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_14/ReadVariableOp(^batch_normalization_14/ReadVariableOp_1&^combined_input/BiasAdd/ReadVariableOp%^combined_input/MatMul/ReadVariableOp!^conv3d_10/BiasAdd/ReadVariableOp ^conv3d_10/Conv3D/ReadVariableOp!^conv3d_11/BiasAdd/ReadVariableOp ^conv3d_11/Conv3D/ReadVariableOp!^conv3d_12/BiasAdd/ReadVariableOp ^conv3d_12/Conv3D/ReadVariableOp!^conv3d_13/BiasAdd/ReadVariableOp ^conv3d_13/Conv3D/ReadVariableOp!^conv3d_14/BiasAdd/ReadVariableOp ^conv3d_14/Conv3D/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp'^ensemble_output/BiasAdd/ReadVariableOp&^ensemble_output/MatMul/ReadVariableOp*"
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
5_dense18/MatMul/ReadVariableOp5_dense18/MatMul/ReadVariableOp2N
%batch_normalization_10/AssignNewValue%batch_normalization_10/AssignNewValue2R
'batch_normalization_10/AssignNewValue_1'batch_normalization_10/AssignNewValue_12p
6batch_normalization_10/FusedBatchNormV3/ReadVariableOp6batch_normalization_10/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_18batch_normalization_10/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_10/ReadVariableOp%batch_normalization_10/ReadVariableOp2R
'batch_normalization_10/ReadVariableOp_1'batch_normalization_10/ReadVariableOp_12N
%batch_normalization_11/AssignNewValue%batch_normalization_11/AssignNewValue2R
'batch_normalization_11/AssignNewValue_1'batch_normalization_11/AssignNewValue_12p
6batch_normalization_11/FusedBatchNormV3/ReadVariableOp6batch_normalization_11/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_18batch_normalization_11/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_11/ReadVariableOp%batch_normalization_11/ReadVariableOp2R
'batch_normalization_11/ReadVariableOp_1'batch_normalization_11/ReadVariableOp_12N
%batch_normalization_12/AssignNewValue%batch_normalization_12/AssignNewValue2R
'batch_normalization_12/AssignNewValue_1'batch_normalization_12/AssignNewValue_12p
6batch_normalization_12/FusedBatchNormV3/ReadVariableOp6batch_normalization_12/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_18batch_normalization_12/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_12/ReadVariableOp%batch_normalization_12/ReadVariableOp2R
'batch_normalization_12/ReadVariableOp_1'batch_normalization_12/ReadVariableOp_12N
%batch_normalization_13/AssignNewValue%batch_normalization_13/AssignNewValue2R
'batch_normalization_13/AssignNewValue_1'batch_normalization_13/AssignNewValue_12p
6batch_normalization_13/FusedBatchNormV3/ReadVariableOp6batch_normalization_13/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_18batch_normalization_13/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_13/ReadVariableOp%batch_normalization_13/ReadVariableOp2R
'batch_normalization_13/ReadVariableOp_1'batch_normalization_13/ReadVariableOp_12N
%batch_normalization_14/AssignNewValue%batch_normalization_14/AssignNewValue2R
'batch_normalization_14/AssignNewValue_1'batch_normalization_14/AssignNewValue_12p
6batch_normalization_14/FusedBatchNormV3/ReadVariableOp6batch_normalization_14/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_14/FusedBatchNormV3/ReadVariableOp_18batch_normalization_14/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_14/ReadVariableOp%batch_normalization_14/ReadVariableOp2R
'batch_normalization_14/ReadVariableOp_1'batch_normalization_14/ReadVariableOp_12N
%combined_input/BiasAdd/ReadVariableOp%combined_input/BiasAdd/ReadVariableOp2L
$combined_input/MatMul/ReadVariableOp$combined_input/MatMul/ReadVariableOp2D
 conv3d_10/BiasAdd/ReadVariableOp conv3d_10/BiasAdd/ReadVariableOp2B
conv3d_10/Conv3D/ReadVariableOpconv3d_10/Conv3D/ReadVariableOp2D
 conv3d_11/BiasAdd/ReadVariableOp conv3d_11/BiasAdd/ReadVariableOp2B
conv3d_11/Conv3D/ReadVariableOpconv3d_11/Conv3D/ReadVariableOp2D
 conv3d_12/BiasAdd/ReadVariableOp conv3d_12/BiasAdd/ReadVariableOp2B
conv3d_12/Conv3D/ReadVariableOpconv3d_12/Conv3D/ReadVariableOp2D
 conv3d_13/BiasAdd/ReadVariableOp conv3d_13/BiasAdd/ReadVariableOp2B
conv3d_13/Conv3D/ReadVariableOpconv3d_13/Conv3D/ReadVariableOp2D
 conv3d_14/BiasAdd/ReadVariableOp conv3d_14/BiasAdd/ReadVariableOp2B
conv3d_14/Conv3D/ReadVariableOpconv3d_14/Conv3D/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2P
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
?	
?
8__inference_batch_normalization_11_layer_call_fn_9336591

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
GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_93349282
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
?
F__inference_conv3d_13_layer_call_and_return_conditional_losses_9336847

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
?
N
2__inference_max_pooling3d_10_layer_call_fn_9336365

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
GPU2*0J 8? *V
fQRO
M__inference_max_pooling3d_10_layer_call_and_return_conditional_losses_93341192
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
?
?
S__inference_batch_normalization_14_layer_call_and_return_conditional_losses_9334741

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
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_9333413

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
?
?
,__inference_ensemble4d_layer_call_fn_9335430

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
G__inference_ensemble4d_layer_call_and_return_conditional_losses_93352372
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
??
?
G__inference_ensemble4d_layer_call_and_return_conditional_losses_9335237

inputs
inputs_1/
conv3d_10_9335118: 
conv3d_10_9335120: ,
batch_normalization_10_9335124: ,
batch_normalization_10_9335126: ,
batch_normalization_10_9335128: ,
batch_normalization_10_9335130: /
conv3d_11_9335133:  
conv3d_11_9335135: ,
batch_normalization_11_9335139: ,
batch_normalization_11_9335141: ,
batch_normalization_11_9335143: ,
batch_normalization_11_9335145: /
conv3d_12_9335148: @
conv3d_12_9335150:@,
batch_normalization_12_9335154:@,
batch_normalization_12_9335156:@,
batch_normalization_12_9335158:@,
batch_normalization_12_9335160:@0
conv3d_13_9335163:@? 
conv3d_13_9335165:	?-
batch_normalization_13_9335169:	?-
batch_normalization_13_9335171:	?-
batch_normalization_13_9335173:	?-
batch_normalization_13_9335175:	?1
conv3d_14_9335178:?? 
conv3d_14_9335180:	?!
dense18_9335184:

dense18_9335186:-
batch_normalization_14_9335189:	?-
batch_normalization_14_9335191:	?-
batch_normalization_14_9335193:	?-
batch_normalization_14_9335195:	?!
dense32_9335198: 
dense32_9335200: !
dense64_9335204: @
dense64_9335206:@#
dense_2_9335209:
??
dense_2_9335211:	?!
dense32_9335214:@ 
dense32_9335216: !
dense18_9335219: 
dense18_9335221:)
combined_input_9335226:	?	$
combined_input_9335228:	)
ensemble_output_9335231:	%
ensemble_output_9335233:
identity??!1_dense18/StatefulPartitionedCall?!2_dense32/StatefulPartitionedCall?!3_dense64/StatefulPartitionedCall?!4_dense32/StatefulPartitionedCall?!5_dense18/StatefulPartitionedCall?.batch_normalization_10/StatefulPartitionedCall?.batch_normalization_11/StatefulPartitionedCall?.batch_normalization_12/StatefulPartitionedCall?.batch_normalization_13/StatefulPartitionedCall?.batch_normalization_14/StatefulPartitionedCall?&combined_input/StatefulPartitionedCall?!conv3d_10/StatefulPartitionedCall?!conv3d_11/StatefulPartitionedCall?!conv3d_12/StatefulPartitionedCall?!conv3d_13/StatefulPartitionedCall?!conv3d_14/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?!dropout_2/StatefulPartitionedCall?'ensemble_output/StatefulPartitionedCall?
!conv3d_10/StatefulPartitionedCallStatefulPartitionedCallinputs_1conv3d_10_9335118conv3d_10_9335120*
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
GPU2*0J 8? *O
fJRH
F__inference_conv3d_10_layer_call_and_return_conditional_losses_93341092#
!conv3d_10/StatefulPartitionedCall?
 max_pooling3d_10/PartitionedCallPartitionedCall*conv3d_10/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *V
fQRO
M__inference_max_pooling3d_10_layer_call_and_return_conditional_losses_93341192"
 max_pooling3d_10/PartitionedCall?
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall)max_pooling3d_10/PartitionedCall:output:0batch_normalization_10_9335124batch_normalization_10_9335126batch_normalization_10_9335128batch_normalization_10_9335130*
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
GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_933498720
.batch_normalization_10/StatefulPartitionedCall?
!conv3d_11/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_10/StatefulPartitionedCall:output:0conv3d_11_9335133conv3d_11_9335135*
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
GPU2*0J 8? *O
fJRH
F__inference_conv3d_11_layer_call_and_return_conditional_losses_93341592#
!conv3d_11/StatefulPartitionedCall?
 max_pooling3d_11/PartitionedCallPartitionedCall*conv3d_11/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *V
fQRO
M__inference_max_pooling3d_11_layer_call_and_return_conditional_losses_93341692"
 max_pooling3d_11/PartitionedCall?
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall)max_pooling3d_11/PartitionedCall:output:0batch_normalization_11_9335139batch_normalization_11_9335141batch_normalization_11_9335143batch_normalization_11_9335145*
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
GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_933492820
.batch_normalization_11/StatefulPartitionedCall?
!conv3d_12/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_11/StatefulPartitionedCall:output:0conv3d_12_9335148conv3d_12_9335150*
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
GPU2*0J 8? *O
fJRH
F__inference_conv3d_12_layer_call_and_return_conditional_losses_93342092#
!conv3d_12/StatefulPartitionedCall?
 max_pooling3d_12/PartitionedCallPartitionedCall*conv3d_12/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *V
fQRO
M__inference_max_pooling3d_12_layer_call_and_return_conditional_losses_93342192"
 max_pooling3d_12/PartitionedCall?
.batch_normalization_12/StatefulPartitionedCallStatefulPartitionedCall)max_pooling3d_12/PartitionedCall:output:0batch_normalization_12_9335154batch_normalization_12_9335156batch_normalization_12_9335158batch_normalization_12_9335160*
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
GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_933486920
.batch_normalization_12/StatefulPartitionedCall?
!conv3d_13/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_12/StatefulPartitionedCall:output:0conv3d_13_9335163conv3d_13_9335165*
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
GPU2*0J 8? *O
fJRH
F__inference_conv3d_13_layer_call_and_return_conditional_losses_93342592#
!conv3d_13/StatefulPartitionedCall?
 max_pooling3d_13/PartitionedCallPartitionedCall*conv3d_13/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *V
fQRO
M__inference_max_pooling3d_13_layer_call_and_return_conditional_losses_93342692"
 max_pooling3d_13/PartitionedCall?
.batch_normalization_13/StatefulPartitionedCallStatefulPartitionedCall)max_pooling3d_13/PartitionedCall:output:0batch_normalization_13_9335169batch_normalization_13_9335171batch_normalization_13_9335173batch_normalization_13_9335175*
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
GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_933481020
.batch_normalization_13/StatefulPartitionedCall?
!conv3d_14/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_13/StatefulPartitionedCall:output:0conv3d_14_9335178conv3d_14_9335180*
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
GPU2*0J 8? *O
fJRH
F__inference_conv3d_14_layer_call_and_return_conditional_losses_93343092#
!conv3d_14/StatefulPartitionedCall?
 max_pooling3d_14/PartitionedCallPartitionedCall*conv3d_14/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *V
fQRO
M__inference_max_pooling3d_14_layer_call_and_return_conditional_losses_93343192"
 max_pooling3d_14/PartitionedCall?
!1_dense18/StatefulPartitionedCallStatefulPartitionedCallinputsdense18_9335184dense18_9335186*
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
F__inference_1_dense18_layer_call_and_return_conditional_losses_93343322#
!1_dense18/StatefulPartitionedCall?
.batch_normalization_14/StatefulPartitionedCallStatefulPartitionedCall)max_pooling3d_14/PartitionedCall:output:0batch_normalization_14_9335189batch_normalization_14_9335191batch_normalization_14_9335193batch_normalization_14_9335195*
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
GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_14_layer_call_and_return_conditional_losses_933474120
.batch_normalization_14/StatefulPartitionedCall?
!2_dense32/StatefulPartitionedCallStatefulPartitionedCall*1_dense18/StatefulPartitionedCall:output:0dense32_9335198dense32_9335200*
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
F__inference_2_dense32_layer_call_and_return_conditional_losses_93343762#
!2_dense32/StatefulPartitionedCall?
*global_average_pooling3d_2/PartitionedCallPartitionedCall7batch_normalization_14/StatefulPartitionedCall:output:0*
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
W__inference_global_average_pooling3d_2_layer_call_and_return_conditional_losses_93343872,
*global_average_pooling3d_2/PartitionedCall?
!3_dense64/StatefulPartitionedCallStatefulPartitionedCall*2_dense32/StatefulPartitionedCall:output:0dense64_9335204dense64_9335206*
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
F__inference_3_dense64_layer_call_and_return_conditional_losses_93344002#
!3_dense64/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling3d_2/PartitionedCall:output:0dense_2_9335209dense_2_9335211*
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
D__inference_dense_2_layer_call_and_return_conditional_losses_93344172!
dense_2/StatefulPartitionedCall?
!4_dense32/StatefulPartitionedCallStatefulPartitionedCall*3_dense64/StatefulPartitionedCall:output:0dense32_9335214dense32_9335216*
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
F__inference_4_dense32_layer_call_and_return_conditional_losses_93344342#
!4_dense32/StatefulPartitionedCall?
!5_dense18/StatefulPartitionedCallStatefulPartitionedCall*4_dense32/StatefulPartitionedCall:output:0dense18_9335219dense18_9335221*
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
F__inference_5_dense18_layer_call_and_return_conditional_losses_93344512#
!5_dense18/StatefulPartitionedCall?
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
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
F__inference_dropout_2_layer_call_and_return_conditional_losses_93346502#
!dropout_2/StatefulPartitionedCall?
concatenate_1/PartitionedCallPartitionedCall*5_dense18/StatefulPartitionedCall:output:0*dropout_2/StatefulPartitionedCall:output:0*
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
J__inference_concatenate_1_layer_call_and_return_conditional_losses_93344712
concatenate_1/PartitionedCall?
&combined_input/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0combined_input_9335226combined_input_9335228*
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
K__inference_combined_input_layer_call_and_return_conditional_losses_93344842(
&combined_input/StatefulPartitionedCall?
'ensemble_output/StatefulPartitionedCallStatefulPartitionedCall/combined_input/StatefulPartitionedCall:output:0ensemble_output_9335231ensemble_output_9335233*
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
L__inference_ensemble_output_layer_call_and_return_conditional_losses_93345012)
'ensemble_output/StatefulPartitionedCall?
IdentityIdentity0ensemble_output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp"^1_dense18/StatefulPartitionedCall"^2_dense32/StatefulPartitionedCall"^3_dense64/StatefulPartitionedCall"^4_dense32/StatefulPartitionedCall"^5_dense18/StatefulPartitionedCall/^batch_normalization_10/StatefulPartitionedCall/^batch_normalization_11/StatefulPartitionedCall/^batch_normalization_12/StatefulPartitionedCall/^batch_normalization_13/StatefulPartitionedCall/^batch_normalization_14/StatefulPartitionedCall'^combined_input/StatefulPartitionedCall"^conv3d_10/StatefulPartitionedCall"^conv3d_11/StatefulPartitionedCall"^conv3d_12/StatefulPartitionedCall"^conv3d_13/StatefulPartitionedCall"^conv3d_14/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall(^ensemble_output/StatefulPartitionedCall*"
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
!5_dense18/StatefulPartitionedCall!5_dense18/StatefulPartitionedCall2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2`
.batch_normalization_12/StatefulPartitionedCall.batch_normalization_12/StatefulPartitionedCall2`
.batch_normalization_13/StatefulPartitionedCall.batch_normalization_13/StatefulPartitionedCall2`
.batch_normalization_14/StatefulPartitionedCall.batch_normalization_14/StatefulPartitionedCall2P
&combined_input/StatefulPartitionedCall&combined_input/StatefulPartitionedCall2F
!conv3d_10/StatefulPartitionedCall!conv3d_10/StatefulPartitionedCall2F
!conv3d_11/StatefulPartitionedCall!conv3d_11/StatefulPartitionedCall2F
!conv3d_12/StatefulPartitionedCall!conv3d_12/StatefulPartitionedCall2F
!conv3d_13/StatefulPartitionedCall!conv3d_13/StatefulPartitionedCall2F
!conv3d_14/StatefulPartitionedCall!conv3d_14/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2R
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
?
L__inference_ensemble_output_layer_call_and_return_conditional_losses_9334501

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
i
M__inference_max_pooling3d_13_layer_call_and_return_conditional_losses_9336862

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
N
2__inference_max_pooling3d_13_layer_call_fn_9336857

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
GPU2*0J 8? *V
fQRO
M__inference_max_pooling3d_13_layer_call_and_return_conditional_losses_93342692
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
?
?
+__inference_2_dense32_layer_call_fn_9337060

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
F__inference_2_dense32_layer_call_and_return_conditional_losses_93343762
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
?
?
+__inference_conv3d_12_layer_call_fn_9336672

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
GPU2*0J 8? *O
fJRH
F__inference_conv3d_12_layer_call_and_return_conditional_losses_93342092
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
?
?
L__inference_ensemble_output_layer_call_and_return_conditional_losses_9337377

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
?
?
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_9334138

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
?
N
2__inference_max_pooling3d_13_layer_call_fn_9336852

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
GPU2*0J 8? *V
fQRO
M__inference_max_pooling3d_13_layer_call_and_return_conditional_losses_93337782
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
?
?
%__inference_signature_wrapper_9335780
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
"__inference__wrapped_model_93333252
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
?
?
F__inference_4_dense32_layer_call_and_return_conditional_losses_9334434

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
?
?
F__inference_conv3d_11_layer_call_and_return_conditional_losses_9336519

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
?
?
+__inference_1_dense18_layer_call_fn_9337020

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
F__inference_1_dense18_layer_call_and_return_conditional_losses_93343322
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
8__inference_batch_normalization_13_layer_call_fn_9336906

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
GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_93342882
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
?
s
W__inference_global_average_pooling3d_2_layer_call_and_return_conditional_losses_9337231

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
?
?
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_9336445

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
?
i
M__inference_max_pooling3d_11_layer_call_and_return_conditional_losses_9333482

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
?
?
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_9336663

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
?
8__inference_batch_normalization_12_layer_call_fn_9336755

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
GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_93348692
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
?
X
<__inference_global_average_pooling3d_2_layer_call_fn_9337220

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
W__inference_global_average_pooling3d_2_layer_call_and_return_conditional_losses_93340752
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
?

?
8__inference_batch_normalization_14_layer_call_fn_9337097

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
GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_14_layer_call_and_return_conditional_losses_93340052
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
?
e
F__inference_dropout_2_layer_call_and_return_conditional_losses_9337324

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
?	
?
8__inference_batch_normalization_12_layer_call_fn_9336729

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
GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_93337092
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
?
?
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_9336499

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
i
M__inference_max_pooling3d_14_layer_call_and_return_conditional_losses_9333926

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
S__inference_batch_normalization_14_layer_call_and_return_conditional_losses_9337141

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
?
N
2__inference_max_pooling3d_11_layer_call_fn_9336529

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
GPU2*0J 8? *V
fQRO
M__inference_max_pooling3d_11_layer_call_and_return_conditional_losses_93341692
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
?
i
M__inference_max_pooling3d_10_layer_call_and_return_conditional_losses_9336370

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
?
?
F__inference_conv3d_14_layer_call_and_return_conditional_losses_9334309

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
?
F__inference_conv3d_14_layer_call_and_return_conditional_losses_9337011

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
?
?
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_9334238

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
?
?
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_9336627

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
?
?
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_9336973

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
?
i
M__inference_max_pooling3d_13_layer_call_and_return_conditional_losses_9334269

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
?
?
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_9333369

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
?
?
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_9333561

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
?
?
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_9333813

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
?
i
M__inference_max_pooling3d_14_layer_call_and_return_conditional_losses_9337046

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
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_9333517

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
?
?
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_9333665

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
?
?
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_9333709

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
?
+__inference_5_dense18_layer_call_fn_9337286

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
F__inference_5_dense18_layer_call_and_return_conditional_losses_93344512
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
?
?
S__inference_batch_normalization_14_layer_call_and_return_conditional_losses_9337177

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
?
i
M__inference_max_pooling3d_12_layer_call_and_return_conditional_losses_9336698

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
8__inference_batch_normalization_14_layer_call_fn_9337123

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
GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_14_layer_call_and_return_conditional_losses_93347412
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
?
?
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_9334810

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
?
s
W__inference_global_average_pooling3d_2_layer_call_and_return_conditional_losses_9334387

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
?
[
/__inference_concatenate_1_layer_call_fn_9337330
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
J__inference_concatenate_1_layer_call_and_return_conditional_losses_93344712
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
?	
?
8__inference_batch_normalization_14_layer_call_fn_9337110

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
GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_14_layer_call_and_return_conditional_losses_93343552
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
8__inference_batch_normalization_10_layer_call_fn_9336427

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
GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_93349872
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
?
F__inference_conv3d_10_layer_call_and_return_conditional_losses_9336355

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
8__inference_batch_normalization_13_layer_call_fn_9336893

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
GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_93338572
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
?
s
W__inference_global_average_pooling3d_2_layer_call_and_return_conditional_losses_9337237

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
?
i
M__inference_max_pooling3d_12_layer_call_and_return_conditional_losses_9334219

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
?
i
M__inference_max_pooling3d_11_layer_call_and_return_conditional_losses_9336534

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
8__inference_batch_normalization_11_layer_call_fn_9336565

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
GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_93335612
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
?
F__inference_3_dense64_layer_call_and_return_conditional_losses_9334400

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
?
?
+__inference_conv3d_13_layer_call_fn_9336836

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
GPU2*0J 8? *O
fJRH
F__inference_conv3d_13_layer_call_and_return_conditional_losses_93342592
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
?
N
2__inference_max_pooling3d_10_layer_call_fn_9336360

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
GPU2*0J 8? *V
fQRO
M__inference_max_pooling3d_10_layer_call_and_return_conditional_losses_93333342
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
i
M__inference_max_pooling3d_11_layer_call_and_return_conditional_losses_9336539

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
?
?
S__inference_batch_normalization_14_layer_call_and_return_conditional_losses_9337195

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
S__inference_batch_normalization_14_layer_call_and_return_conditional_losses_9334005

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
??
?
G__inference_ensemble4d_layer_call_and_return_conditional_losses_9335553

svm_inputs
img3d_inputs/
conv3d_10_9335434: 
conv3d_10_9335436: ,
batch_normalization_10_9335440: ,
batch_normalization_10_9335442: ,
batch_normalization_10_9335444: ,
batch_normalization_10_9335446: /
conv3d_11_9335449:  
conv3d_11_9335451: ,
batch_normalization_11_9335455: ,
batch_normalization_11_9335457: ,
batch_normalization_11_9335459: ,
batch_normalization_11_9335461: /
conv3d_12_9335464: @
conv3d_12_9335466:@,
batch_normalization_12_9335470:@,
batch_normalization_12_9335472:@,
batch_normalization_12_9335474:@,
batch_normalization_12_9335476:@0
conv3d_13_9335479:@? 
conv3d_13_9335481:	?-
batch_normalization_13_9335485:	?-
batch_normalization_13_9335487:	?-
batch_normalization_13_9335489:	?-
batch_normalization_13_9335491:	?1
conv3d_14_9335494:?? 
conv3d_14_9335496:	?!
dense18_9335500:

dense18_9335502:-
batch_normalization_14_9335505:	?-
batch_normalization_14_9335507:	?-
batch_normalization_14_9335509:	?-
batch_normalization_14_9335511:	?!
dense32_9335514: 
dense32_9335516: !
dense64_9335520: @
dense64_9335522:@#
dense_2_9335525:
??
dense_2_9335527:	?!
dense32_9335530:@ 
dense32_9335532: !
dense18_9335535: 
dense18_9335537:)
combined_input_9335542:	?	$
combined_input_9335544:	)
ensemble_output_9335547:	%
ensemble_output_9335549:
identity??!1_dense18/StatefulPartitionedCall?!2_dense32/StatefulPartitionedCall?!3_dense64/StatefulPartitionedCall?!4_dense32/StatefulPartitionedCall?!5_dense18/StatefulPartitionedCall?.batch_normalization_10/StatefulPartitionedCall?.batch_normalization_11/StatefulPartitionedCall?.batch_normalization_12/StatefulPartitionedCall?.batch_normalization_13/StatefulPartitionedCall?.batch_normalization_14/StatefulPartitionedCall?&combined_input/StatefulPartitionedCall?!conv3d_10/StatefulPartitionedCall?!conv3d_11/StatefulPartitionedCall?!conv3d_12/StatefulPartitionedCall?!conv3d_13/StatefulPartitionedCall?!conv3d_14/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?'ensemble_output/StatefulPartitionedCall?
!conv3d_10/StatefulPartitionedCallStatefulPartitionedCallimg3d_inputsconv3d_10_9335434conv3d_10_9335436*
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
GPU2*0J 8? *O
fJRH
F__inference_conv3d_10_layer_call_and_return_conditional_losses_93341092#
!conv3d_10/StatefulPartitionedCall?
 max_pooling3d_10/PartitionedCallPartitionedCall*conv3d_10/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *V
fQRO
M__inference_max_pooling3d_10_layer_call_and_return_conditional_losses_93341192"
 max_pooling3d_10/PartitionedCall?
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall)max_pooling3d_10/PartitionedCall:output:0batch_normalization_10_9335440batch_normalization_10_9335442batch_normalization_10_9335444batch_normalization_10_9335446*
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
GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_933413820
.batch_normalization_10/StatefulPartitionedCall?
!conv3d_11/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_10/StatefulPartitionedCall:output:0conv3d_11_9335449conv3d_11_9335451*
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
GPU2*0J 8? *O
fJRH
F__inference_conv3d_11_layer_call_and_return_conditional_losses_93341592#
!conv3d_11/StatefulPartitionedCall?
 max_pooling3d_11/PartitionedCallPartitionedCall*conv3d_11/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *V
fQRO
M__inference_max_pooling3d_11_layer_call_and_return_conditional_losses_93341692"
 max_pooling3d_11/PartitionedCall?
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall)max_pooling3d_11/PartitionedCall:output:0batch_normalization_11_9335455batch_normalization_11_9335457batch_normalization_11_9335459batch_normalization_11_9335461*
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
GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_933418820
.batch_normalization_11/StatefulPartitionedCall?
!conv3d_12/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_11/StatefulPartitionedCall:output:0conv3d_12_9335464conv3d_12_9335466*
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
GPU2*0J 8? *O
fJRH
F__inference_conv3d_12_layer_call_and_return_conditional_losses_93342092#
!conv3d_12/StatefulPartitionedCall?
 max_pooling3d_12/PartitionedCallPartitionedCall*conv3d_12/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *V
fQRO
M__inference_max_pooling3d_12_layer_call_and_return_conditional_losses_93342192"
 max_pooling3d_12/PartitionedCall?
.batch_normalization_12/StatefulPartitionedCallStatefulPartitionedCall)max_pooling3d_12/PartitionedCall:output:0batch_normalization_12_9335470batch_normalization_12_9335472batch_normalization_12_9335474batch_normalization_12_9335476*
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
GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_933423820
.batch_normalization_12/StatefulPartitionedCall?
!conv3d_13/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_12/StatefulPartitionedCall:output:0conv3d_13_9335479conv3d_13_9335481*
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
GPU2*0J 8? *O
fJRH
F__inference_conv3d_13_layer_call_and_return_conditional_losses_93342592#
!conv3d_13/StatefulPartitionedCall?
 max_pooling3d_13/PartitionedCallPartitionedCall*conv3d_13/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *V
fQRO
M__inference_max_pooling3d_13_layer_call_and_return_conditional_losses_93342692"
 max_pooling3d_13/PartitionedCall?
.batch_normalization_13/StatefulPartitionedCallStatefulPartitionedCall)max_pooling3d_13/PartitionedCall:output:0batch_normalization_13_9335485batch_normalization_13_9335487batch_normalization_13_9335489batch_normalization_13_9335491*
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
GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_933428820
.batch_normalization_13/StatefulPartitionedCall?
!conv3d_14/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_13/StatefulPartitionedCall:output:0conv3d_14_9335494conv3d_14_9335496*
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
GPU2*0J 8? *O
fJRH
F__inference_conv3d_14_layer_call_and_return_conditional_losses_93343092#
!conv3d_14/StatefulPartitionedCall?
 max_pooling3d_14/PartitionedCallPartitionedCall*conv3d_14/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *V
fQRO
M__inference_max_pooling3d_14_layer_call_and_return_conditional_losses_93343192"
 max_pooling3d_14/PartitionedCall?
!1_dense18/StatefulPartitionedCallStatefulPartitionedCall
svm_inputsdense18_9335500dense18_9335502*
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
F__inference_1_dense18_layer_call_and_return_conditional_losses_93343322#
!1_dense18/StatefulPartitionedCall?
.batch_normalization_14/StatefulPartitionedCallStatefulPartitionedCall)max_pooling3d_14/PartitionedCall:output:0batch_normalization_14_9335505batch_normalization_14_9335507batch_normalization_14_9335509batch_normalization_14_9335511*
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
GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_14_layer_call_and_return_conditional_losses_933435520
.batch_normalization_14/StatefulPartitionedCall?
!2_dense32/StatefulPartitionedCallStatefulPartitionedCall*1_dense18/StatefulPartitionedCall:output:0dense32_9335514dense32_9335516*
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
F__inference_2_dense32_layer_call_and_return_conditional_losses_93343762#
!2_dense32/StatefulPartitionedCall?
*global_average_pooling3d_2/PartitionedCallPartitionedCall7batch_normalization_14/StatefulPartitionedCall:output:0*
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
W__inference_global_average_pooling3d_2_layer_call_and_return_conditional_losses_93343872,
*global_average_pooling3d_2/PartitionedCall?
!3_dense64/StatefulPartitionedCallStatefulPartitionedCall*2_dense32/StatefulPartitionedCall:output:0dense64_9335520dense64_9335522*
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
F__inference_3_dense64_layer_call_and_return_conditional_losses_93344002#
!3_dense64/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling3d_2/PartitionedCall:output:0dense_2_9335525dense_2_9335527*
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
D__inference_dense_2_layer_call_and_return_conditional_losses_93344172!
dense_2/StatefulPartitionedCall?
!4_dense32/StatefulPartitionedCallStatefulPartitionedCall*3_dense64/StatefulPartitionedCall:output:0dense32_9335530dense32_9335532*
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
F__inference_4_dense32_layer_call_and_return_conditional_losses_93344342#
!4_dense32/StatefulPartitionedCall?
!5_dense18/StatefulPartitionedCallStatefulPartitionedCall*4_dense32/StatefulPartitionedCall:output:0dense18_9335535dense18_9335537*
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
F__inference_5_dense18_layer_call_and_return_conditional_losses_93344512#
!5_dense18/StatefulPartitionedCall?
dropout_2/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
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
F__inference_dropout_2_layer_call_and_return_conditional_losses_93344622
dropout_2/PartitionedCall?
concatenate_1/PartitionedCallPartitionedCall*5_dense18/StatefulPartitionedCall:output:0"dropout_2/PartitionedCall:output:0*
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
J__inference_concatenate_1_layer_call_and_return_conditional_losses_93344712
concatenate_1/PartitionedCall?
&combined_input/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0combined_input_9335542combined_input_9335544*
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
K__inference_combined_input_layer_call_and_return_conditional_losses_93344842(
&combined_input/StatefulPartitionedCall?
'ensemble_output/StatefulPartitionedCallStatefulPartitionedCall/combined_input/StatefulPartitionedCall:output:0ensemble_output_9335547ensemble_output_9335549*
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
L__inference_ensemble_output_layer_call_and_return_conditional_losses_93345012)
'ensemble_output/StatefulPartitionedCall?
IdentityIdentity0ensemble_output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp"^1_dense18/StatefulPartitionedCall"^2_dense32/StatefulPartitionedCall"^3_dense64/StatefulPartitionedCall"^4_dense32/StatefulPartitionedCall"^5_dense18/StatefulPartitionedCall/^batch_normalization_10/StatefulPartitionedCall/^batch_normalization_11/StatefulPartitionedCall/^batch_normalization_12/StatefulPartitionedCall/^batch_normalization_13/StatefulPartitionedCall/^batch_normalization_14/StatefulPartitionedCall'^combined_input/StatefulPartitionedCall"^conv3d_10/StatefulPartitionedCall"^conv3d_11/StatefulPartitionedCall"^conv3d_12/StatefulPartitionedCall"^conv3d_13/StatefulPartitionedCall"^conv3d_14/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall(^ensemble_output/StatefulPartitionedCall*"
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
!5_dense18/StatefulPartitionedCall!5_dense18/StatefulPartitionedCall2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2`
.batch_normalization_12/StatefulPartitionedCall.batch_normalization_12/StatefulPartitionedCall2`
.batch_normalization_13/StatefulPartitionedCall.batch_normalization_13/StatefulPartitionedCall2`
.batch_normalization_14/StatefulPartitionedCall.batch_normalization_14/StatefulPartitionedCall2P
&combined_input/StatefulPartitionedCall&combined_input/StatefulPartitionedCall2F
!conv3d_10/StatefulPartitionedCall!conv3d_10/StatefulPartitionedCall2F
!conv3d_11/StatefulPartitionedCall!conv3d_11/StatefulPartitionedCall2F
!conv3d_12/StatefulPartitionedCall!conv3d_12/StatefulPartitionedCall2F
!conv3d_13/StatefulPartitionedCall!conv3d_13/StatefulPartitionedCall2F
!conv3d_14/StatefulPartitionedCall!conv3d_14/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2R
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
?
?
F__inference_2_dense32_layer_call_and_return_conditional_losses_9337071

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
s
W__inference_global_average_pooling3d_2_layer_call_and_return_conditional_losses_9334075

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
?
i
M__inference_max_pooling3d_14_layer_call_and_return_conditional_losses_9334319

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
??
?
G__inference_ensemble4d_layer_call_and_return_conditional_losses_9335676

svm_inputs
img3d_inputs/
conv3d_10_9335557: 
conv3d_10_9335559: ,
batch_normalization_10_9335563: ,
batch_normalization_10_9335565: ,
batch_normalization_10_9335567: ,
batch_normalization_10_9335569: /
conv3d_11_9335572:  
conv3d_11_9335574: ,
batch_normalization_11_9335578: ,
batch_normalization_11_9335580: ,
batch_normalization_11_9335582: ,
batch_normalization_11_9335584: /
conv3d_12_9335587: @
conv3d_12_9335589:@,
batch_normalization_12_9335593:@,
batch_normalization_12_9335595:@,
batch_normalization_12_9335597:@,
batch_normalization_12_9335599:@0
conv3d_13_9335602:@? 
conv3d_13_9335604:	?-
batch_normalization_13_9335608:	?-
batch_normalization_13_9335610:	?-
batch_normalization_13_9335612:	?-
batch_normalization_13_9335614:	?1
conv3d_14_9335617:?? 
conv3d_14_9335619:	?!
dense18_9335623:

dense18_9335625:-
batch_normalization_14_9335628:	?-
batch_normalization_14_9335630:	?-
batch_normalization_14_9335632:	?-
batch_normalization_14_9335634:	?!
dense32_9335637: 
dense32_9335639: !
dense64_9335643: @
dense64_9335645:@#
dense_2_9335648:
??
dense_2_9335650:	?!
dense32_9335653:@ 
dense32_9335655: !
dense18_9335658: 
dense18_9335660:)
combined_input_9335665:	?	$
combined_input_9335667:	)
ensemble_output_9335670:	%
ensemble_output_9335672:
identity??!1_dense18/StatefulPartitionedCall?!2_dense32/StatefulPartitionedCall?!3_dense64/StatefulPartitionedCall?!4_dense32/StatefulPartitionedCall?!5_dense18/StatefulPartitionedCall?.batch_normalization_10/StatefulPartitionedCall?.batch_normalization_11/StatefulPartitionedCall?.batch_normalization_12/StatefulPartitionedCall?.batch_normalization_13/StatefulPartitionedCall?.batch_normalization_14/StatefulPartitionedCall?&combined_input/StatefulPartitionedCall?!conv3d_10/StatefulPartitionedCall?!conv3d_11/StatefulPartitionedCall?!conv3d_12/StatefulPartitionedCall?!conv3d_13/StatefulPartitionedCall?!conv3d_14/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?!dropout_2/StatefulPartitionedCall?'ensemble_output/StatefulPartitionedCall?
!conv3d_10/StatefulPartitionedCallStatefulPartitionedCallimg3d_inputsconv3d_10_9335557conv3d_10_9335559*
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
GPU2*0J 8? *O
fJRH
F__inference_conv3d_10_layer_call_and_return_conditional_losses_93341092#
!conv3d_10/StatefulPartitionedCall?
 max_pooling3d_10/PartitionedCallPartitionedCall*conv3d_10/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *V
fQRO
M__inference_max_pooling3d_10_layer_call_and_return_conditional_losses_93341192"
 max_pooling3d_10/PartitionedCall?
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall)max_pooling3d_10/PartitionedCall:output:0batch_normalization_10_9335563batch_normalization_10_9335565batch_normalization_10_9335567batch_normalization_10_9335569*
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
GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_933498720
.batch_normalization_10/StatefulPartitionedCall?
!conv3d_11/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_10/StatefulPartitionedCall:output:0conv3d_11_9335572conv3d_11_9335574*
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
GPU2*0J 8? *O
fJRH
F__inference_conv3d_11_layer_call_and_return_conditional_losses_93341592#
!conv3d_11/StatefulPartitionedCall?
 max_pooling3d_11/PartitionedCallPartitionedCall*conv3d_11/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *V
fQRO
M__inference_max_pooling3d_11_layer_call_and_return_conditional_losses_93341692"
 max_pooling3d_11/PartitionedCall?
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall)max_pooling3d_11/PartitionedCall:output:0batch_normalization_11_9335578batch_normalization_11_9335580batch_normalization_11_9335582batch_normalization_11_9335584*
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
GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_933492820
.batch_normalization_11/StatefulPartitionedCall?
!conv3d_12/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_11/StatefulPartitionedCall:output:0conv3d_12_9335587conv3d_12_9335589*
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
GPU2*0J 8? *O
fJRH
F__inference_conv3d_12_layer_call_and_return_conditional_losses_93342092#
!conv3d_12/StatefulPartitionedCall?
 max_pooling3d_12/PartitionedCallPartitionedCall*conv3d_12/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *V
fQRO
M__inference_max_pooling3d_12_layer_call_and_return_conditional_losses_93342192"
 max_pooling3d_12/PartitionedCall?
.batch_normalization_12/StatefulPartitionedCallStatefulPartitionedCall)max_pooling3d_12/PartitionedCall:output:0batch_normalization_12_9335593batch_normalization_12_9335595batch_normalization_12_9335597batch_normalization_12_9335599*
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
GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_933486920
.batch_normalization_12/StatefulPartitionedCall?
!conv3d_13/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_12/StatefulPartitionedCall:output:0conv3d_13_9335602conv3d_13_9335604*
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
GPU2*0J 8? *O
fJRH
F__inference_conv3d_13_layer_call_and_return_conditional_losses_93342592#
!conv3d_13/StatefulPartitionedCall?
 max_pooling3d_13/PartitionedCallPartitionedCall*conv3d_13/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *V
fQRO
M__inference_max_pooling3d_13_layer_call_and_return_conditional_losses_93342692"
 max_pooling3d_13/PartitionedCall?
.batch_normalization_13/StatefulPartitionedCallStatefulPartitionedCall)max_pooling3d_13/PartitionedCall:output:0batch_normalization_13_9335608batch_normalization_13_9335610batch_normalization_13_9335612batch_normalization_13_9335614*
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
GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_933481020
.batch_normalization_13/StatefulPartitionedCall?
!conv3d_14/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_13/StatefulPartitionedCall:output:0conv3d_14_9335617conv3d_14_9335619*
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
GPU2*0J 8? *O
fJRH
F__inference_conv3d_14_layer_call_and_return_conditional_losses_93343092#
!conv3d_14/StatefulPartitionedCall?
 max_pooling3d_14/PartitionedCallPartitionedCall*conv3d_14/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *V
fQRO
M__inference_max_pooling3d_14_layer_call_and_return_conditional_losses_93343192"
 max_pooling3d_14/PartitionedCall?
!1_dense18/StatefulPartitionedCallStatefulPartitionedCall
svm_inputsdense18_9335623dense18_9335625*
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
F__inference_1_dense18_layer_call_and_return_conditional_losses_93343322#
!1_dense18/StatefulPartitionedCall?
.batch_normalization_14/StatefulPartitionedCallStatefulPartitionedCall)max_pooling3d_14/PartitionedCall:output:0batch_normalization_14_9335628batch_normalization_14_9335630batch_normalization_14_9335632batch_normalization_14_9335634*
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
GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_14_layer_call_and_return_conditional_losses_933474120
.batch_normalization_14/StatefulPartitionedCall?
!2_dense32/StatefulPartitionedCallStatefulPartitionedCall*1_dense18/StatefulPartitionedCall:output:0dense32_9335637dense32_9335639*
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
F__inference_2_dense32_layer_call_and_return_conditional_losses_93343762#
!2_dense32/StatefulPartitionedCall?
*global_average_pooling3d_2/PartitionedCallPartitionedCall7batch_normalization_14/StatefulPartitionedCall:output:0*
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
W__inference_global_average_pooling3d_2_layer_call_and_return_conditional_losses_93343872,
*global_average_pooling3d_2/PartitionedCall?
!3_dense64/StatefulPartitionedCallStatefulPartitionedCall*2_dense32/StatefulPartitionedCall:output:0dense64_9335643dense64_9335645*
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
F__inference_3_dense64_layer_call_and_return_conditional_losses_93344002#
!3_dense64/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling3d_2/PartitionedCall:output:0dense_2_9335648dense_2_9335650*
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
D__inference_dense_2_layer_call_and_return_conditional_losses_93344172!
dense_2/StatefulPartitionedCall?
!4_dense32/StatefulPartitionedCallStatefulPartitionedCall*3_dense64/StatefulPartitionedCall:output:0dense32_9335653dense32_9335655*
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
F__inference_4_dense32_layer_call_and_return_conditional_losses_93344342#
!4_dense32/StatefulPartitionedCall?
!5_dense18/StatefulPartitionedCallStatefulPartitionedCall*4_dense32/StatefulPartitionedCall:output:0dense18_9335658dense18_9335660*
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
F__inference_5_dense18_layer_call_and_return_conditional_losses_93344512#
!5_dense18/StatefulPartitionedCall?
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
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
F__inference_dropout_2_layer_call_and_return_conditional_losses_93346502#
!dropout_2/StatefulPartitionedCall?
concatenate_1/PartitionedCallPartitionedCall*5_dense18/StatefulPartitionedCall:output:0*dropout_2/StatefulPartitionedCall:output:0*
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
J__inference_concatenate_1_layer_call_and_return_conditional_losses_93344712
concatenate_1/PartitionedCall?
&combined_input/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0combined_input_9335665combined_input_9335667*
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
K__inference_combined_input_layer_call_and_return_conditional_losses_93344842(
&combined_input/StatefulPartitionedCall?
'ensemble_output/StatefulPartitionedCallStatefulPartitionedCall/combined_input/StatefulPartitionedCall:output:0ensemble_output_9335670ensemble_output_9335672*
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
L__inference_ensemble_output_layer_call_and_return_conditional_losses_93345012)
'ensemble_output/StatefulPartitionedCall?
IdentityIdentity0ensemble_output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp"^1_dense18/StatefulPartitionedCall"^2_dense32/StatefulPartitionedCall"^3_dense64/StatefulPartitionedCall"^4_dense32/StatefulPartitionedCall"^5_dense18/StatefulPartitionedCall/^batch_normalization_10/StatefulPartitionedCall/^batch_normalization_11/StatefulPartitionedCall/^batch_normalization_12/StatefulPartitionedCall/^batch_normalization_13/StatefulPartitionedCall/^batch_normalization_14/StatefulPartitionedCall'^combined_input/StatefulPartitionedCall"^conv3d_10/StatefulPartitionedCall"^conv3d_11/StatefulPartitionedCall"^conv3d_12/StatefulPartitionedCall"^conv3d_13/StatefulPartitionedCall"^conv3d_14/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall(^ensemble_output/StatefulPartitionedCall*"
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
!5_dense18/StatefulPartitionedCall!5_dense18/StatefulPartitionedCall2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2`
.batch_normalization_12/StatefulPartitionedCall.batch_normalization_12/StatefulPartitionedCall2`
.batch_normalization_13/StatefulPartitionedCall.batch_normalization_13/StatefulPartitionedCall2`
.batch_normalization_14/StatefulPartitionedCall.batch_normalization_14/StatefulPartitionedCall2P
&combined_input/StatefulPartitionedCall&combined_input/StatefulPartitionedCall2F
!conv3d_10/StatefulPartitionedCall!conv3d_10/StatefulPartitionedCall2F
!conv3d_11/StatefulPartitionedCall!conv3d_11/StatefulPartitionedCall2F
!conv3d_12/StatefulPartitionedCall!conv3d_12/StatefulPartitionedCall2F
!conv3d_13/StatefulPartitionedCall!conv3d_13/StatefulPartitionedCall2F
!conv3d_14/StatefulPartitionedCall!conv3d_14/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2R
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
,__inference_ensemble4d_layer_call_fn_9335878
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
G__inference_ensemble4d_layer_call_and_return_conditional_losses_93345082
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
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_9336773

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
?
N
2__inference_max_pooling3d_11_layer_call_fn_9336524

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
GPU2*0J 8? *V
fQRO
M__inference_max_pooling3d_11_layer_call_and_return_conditional_losses_93334822
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
?
?
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_9334188

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
?
i
M__inference_max_pooling3d_14_layer_call_and_return_conditional_losses_9337051

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
?	
?
8__inference_batch_normalization_10_layer_call_fn_9336388

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
GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_93333692
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
?
i
M__inference_max_pooling3d_10_layer_call_and_return_conditional_losses_9336375

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
F__inference_2_dense32_layer_call_and_return_conditional_losses_9334376

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
?
d
F__inference_dropout_2_layer_call_and_return_conditional_losses_9337312

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
?
e
F__inference_dropout_2_layer_call_and_return_conditional_losses_9334650

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
?	
?
8__inference_batch_normalization_11_layer_call_fn_9336552

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
GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_93335172
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
?
?
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_9336991

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
?
X
<__inference_global_average_pooling3d_2_layer_call_fn_9337225

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
W__inference_global_average_pooling3d_2_layer_call_and_return_conditional_losses_93343872
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
?
i
M__inference_max_pooling3d_11_layer_call_and_return_conditional_losses_9334169

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
?
?
K__inference_combined_input_layer_call_and_return_conditional_losses_9334484

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
?
i
M__inference_max_pooling3d_10_layer_call_and_return_conditional_losses_9334119

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
?
?
S__inference_batch_normalization_14_layer_call_and_return_conditional_losses_9334355

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
??
?
G__inference_ensemble4d_layer_call_and_return_conditional_losses_9334508

inputs
inputs_1/
conv3d_10_9334110: 
conv3d_10_9334112: ,
batch_normalization_10_9334139: ,
batch_normalization_10_9334141: ,
batch_normalization_10_9334143: ,
batch_normalization_10_9334145: /
conv3d_11_9334160:  
conv3d_11_9334162: ,
batch_normalization_11_9334189: ,
batch_normalization_11_9334191: ,
batch_normalization_11_9334193: ,
batch_normalization_11_9334195: /
conv3d_12_9334210: @
conv3d_12_9334212:@,
batch_normalization_12_9334239:@,
batch_normalization_12_9334241:@,
batch_normalization_12_9334243:@,
batch_normalization_12_9334245:@0
conv3d_13_9334260:@? 
conv3d_13_9334262:	?-
batch_normalization_13_9334289:	?-
batch_normalization_13_9334291:	?-
batch_normalization_13_9334293:	?-
batch_normalization_13_9334295:	?1
conv3d_14_9334310:?? 
conv3d_14_9334312:	?!
dense18_9334333:

dense18_9334335:-
batch_normalization_14_9334356:	?-
batch_normalization_14_9334358:	?-
batch_normalization_14_9334360:	?-
batch_normalization_14_9334362:	?!
dense32_9334377: 
dense32_9334379: !
dense64_9334401: @
dense64_9334403:@#
dense_2_9334418:
??
dense_2_9334420:	?!
dense32_9334435:@ 
dense32_9334437: !
dense18_9334452: 
dense18_9334454:)
combined_input_9334485:	?	$
combined_input_9334487:	)
ensemble_output_9334502:	%
ensemble_output_9334504:
identity??!1_dense18/StatefulPartitionedCall?!2_dense32/StatefulPartitionedCall?!3_dense64/StatefulPartitionedCall?!4_dense32/StatefulPartitionedCall?!5_dense18/StatefulPartitionedCall?.batch_normalization_10/StatefulPartitionedCall?.batch_normalization_11/StatefulPartitionedCall?.batch_normalization_12/StatefulPartitionedCall?.batch_normalization_13/StatefulPartitionedCall?.batch_normalization_14/StatefulPartitionedCall?&combined_input/StatefulPartitionedCall?!conv3d_10/StatefulPartitionedCall?!conv3d_11/StatefulPartitionedCall?!conv3d_12/StatefulPartitionedCall?!conv3d_13/StatefulPartitionedCall?!conv3d_14/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?'ensemble_output/StatefulPartitionedCall?
!conv3d_10/StatefulPartitionedCallStatefulPartitionedCallinputs_1conv3d_10_9334110conv3d_10_9334112*
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
GPU2*0J 8? *O
fJRH
F__inference_conv3d_10_layer_call_and_return_conditional_losses_93341092#
!conv3d_10/StatefulPartitionedCall?
 max_pooling3d_10/PartitionedCallPartitionedCall*conv3d_10/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *V
fQRO
M__inference_max_pooling3d_10_layer_call_and_return_conditional_losses_93341192"
 max_pooling3d_10/PartitionedCall?
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall)max_pooling3d_10/PartitionedCall:output:0batch_normalization_10_9334139batch_normalization_10_9334141batch_normalization_10_9334143batch_normalization_10_9334145*
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
GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_933413820
.batch_normalization_10/StatefulPartitionedCall?
!conv3d_11/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_10/StatefulPartitionedCall:output:0conv3d_11_9334160conv3d_11_9334162*
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
GPU2*0J 8? *O
fJRH
F__inference_conv3d_11_layer_call_and_return_conditional_losses_93341592#
!conv3d_11/StatefulPartitionedCall?
 max_pooling3d_11/PartitionedCallPartitionedCall*conv3d_11/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *V
fQRO
M__inference_max_pooling3d_11_layer_call_and_return_conditional_losses_93341692"
 max_pooling3d_11/PartitionedCall?
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall)max_pooling3d_11/PartitionedCall:output:0batch_normalization_11_9334189batch_normalization_11_9334191batch_normalization_11_9334193batch_normalization_11_9334195*
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
GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_933418820
.batch_normalization_11/StatefulPartitionedCall?
!conv3d_12/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_11/StatefulPartitionedCall:output:0conv3d_12_9334210conv3d_12_9334212*
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
GPU2*0J 8? *O
fJRH
F__inference_conv3d_12_layer_call_and_return_conditional_losses_93342092#
!conv3d_12/StatefulPartitionedCall?
 max_pooling3d_12/PartitionedCallPartitionedCall*conv3d_12/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *V
fQRO
M__inference_max_pooling3d_12_layer_call_and_return_conditional_losses_93342192"
 max_pooling3d_12/PartitionedCall?
.batch_normalization_12/StatefulPartitionedCallStatefulPartitionedCall)max_pooling3d_12/PartitionedCall:output:0batch_normalization_12_9334239batch_normalization_12_9334241batch_normalization_12_9334243batch_normalization_12_9334245*
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
GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_933423820
.batch_normalization_12/StatefulPartitionedCall?
!conv3d_13/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_12/StatefulPartitionedCall:output:0conv3d_13_9334260conv3d_13_9334262*
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
GPU2*0J 8? *O
fJRH
F__inference_conv3d_13_layer_call_and_return_conditional_losses_93342592#
!conv3d_13/StatefulPartitionedCall?
 max_pooling3d_13/PartitionedCallPartitionedCall*conv3d_13/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *V
fQRO
M__inference_max_pooling3d_13_layer_call_and_return_conditional_losses_93342692"
 max_pooling3d_13/PartitionedCall?
.batch_normalization_13/StatefulPartitionedCallStatefulPartitionedCall)max_pooling3d_13/PartitionedCall:output:0batch_normalization_13_9334289batch_normalization_13_9334291batch_normalization_13_9334293batch_normalization_13_9334295*
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
GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_933428820
.batch_normalization_13/StatefulPartitionedCall?
!conv3d_14/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_13/StatefulPartitionedCall:output:0conv3d_14_9334310conv3d_14_9334312*
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
GPU2*0J 8? *O
fJRH
F__inference_conv3d_14_layer_call_and_return_conditional_losses_93343092#
!conv3d_14/StatefulPartitionedCall?
 max_pooling3d_14/PartitionedCallPartitionedCall*conv3d_14/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *V
fQRO
M__inference_max_pooling3d_14_layer_call_and_return_conditional_losses_93343192"
 max_pooling3d_14/PartitionedCall?
!1_dense18/StatefulPartitionedCallStatefulPartitionedCallinputsdense18_9334333dense18_9334335*
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
F__inference_1_dense18_layer_call_and_return_conditional_losses_93343322#
!1_dense18/StatefulPartitionedCall?
.batch_normalization_14/StatefulPartitionedCallStatefulPartitionedCall)max_pooling3d_14/PartitionedCall:output:0batch_normalization_14_9334356batch_normalization_14_9334358batch_normalization_14_9334360batch_normalization_14_9334362*
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
GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_14_layer_call_and_return_conditional_losses_933435520
.batch_normalization_14/StatefulPartitionedCall?
!2_dense32/StatefulPartitionedCallStatefulPartitionedCall*1_dense18/StatefulPartitionedCall:output:0dense32_9334377dense32_9334379*
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
F__inference_2_dense32_layer_call_and_return_conditional_losses_93343762#
!2_dense32/StatefulPartitionedCall?
*global_average_pooling3d_2/PartitionedCallPartitionedCall7batch_normalization_14/StatefulPartitionedCall:output:0*
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
W__inference_global_average_pooling3d_2_layer_call_and_return_conditional_losses_93343872,
*global_average_pooling3d_2/PartitionedCall?
!3_dense64/StatefulPartitionedCallStatefulPartitionedCall*2_dense32/StatefulPartitionedCall:output:0dense64_9334401dense64_9334403*
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
F__inference_3_dense64_layer_call_and_return_conditional_losses_93344002#
!3_dense64/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling3d_2/PartitionedCall:output:0dense_2_9334418dense_2_9334420*
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
D__inference_dense_2_layer_call_and_return_conditional_losses_93344172!
dense_2/StatefulPartitionedCall?
!4_dense32/StatefulPartitionedCallStatefulPartitionedCall*3_dense64/StatefulPartitionedCall:output:0dense32_9334435dense32_9334437*
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
F__inference_4_dense32_layer_call_and_return_conditional_losses_93344342#
!4_dense32/StatefulPartitionedCall?
!5_dense18/StatefulPartitionedCallStatefulPartitionedCall*4_dense32/StatefulPartitionedCall:output:0dense18_9334452dense18_9334454*
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
F__inference_5_dense18_layer_call_and_return_conditional_losses_93344512#
!5_dense18/StatefulPartitionedCall?
dropout_2/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
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
F__inference_dropout_2_layer_call_and_return_conditional_losses_93344622
dropout_2/PartitionedCall?
concatenate_1/PartitionedCallPartitionedCall*5_dense18/StatefulPartitionedCall:output:0"dropout_2/PartitionedCall:output:0*
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
J__inference_concatenate_1_layer_call_and_return_conditional_losses_93344712
concatenate_1/PartitionedCall?
&combined_input/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0combined_input_9334485combined_input_9334487*
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
K__inference_combined_input_layer_call_and_return_conditional_losses_93344842(
&combined_input/StatefulPartitionedCall?
'ensemble_output/StatefulPartitionedCallStatefulPartitionedCall/combined_input/StatefulPartitionedCall:output:0ensemble_output_9334502ensemble_output_9334504*
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
L__inference_ensemble_output_layer_call_and_return_conditional_losses_93345012)
'ensemble_output/StatefulPartitionedCall?
IdentityIdentity0ensemble_output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp"^1_dense18/StatefulPartitionedCall"^2_dense32/StatefulPartitionedCall"^3_dense64/StatefulPartitionedCall"^4_dense32/StatefulPartitionedCall"^5_dense18/StatefulPartitionedCall/^batch_normalization_10/StatefulPartitionedCall/^batch_normalization_11/StatefulPartitionedCall/^batch_normalization_12/StatefulPartitionedCall/^batch_normalization_13/StatefulPartitionedCall/^batch_normalization_14/StatefulPartitionedCall'^combined_input/StatefulPartitionedCall"^conv3d_10/StatefulPartitionedCall"^conv3d_11/StatefulPartitionedCall"^conv3d_12/StatefulPartitionedCall"^conv3d_13/StatefulPartitionedCall"^conv3d_14/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall(^ensemble_output/StatefulPartitionedCall*"
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
!5_dense18/StatefulPartitionedCall!5_dense18/StatefulPartitionedCall2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2`
.batch_normalization_12/StatefulPartitionedCall.batch_normalization_12/StatefulPartitionedCall2`
.batch_normalization_13/StatefulPartitionedCall.batch_normalization_13/StatefulPartitionedCall2`
.batch_normalization_14/StatefulPartitionedCall.batch_normalization_14/StatefulPartitionedCall2P
&combined_input/StatefulPartitionedCall&combined_input/StatefulPartitionedCall2F
!conv3d_10/StatefulPartitionedCall!conv3d_10/StatefulPartitionedCall2F
!conv3d_11/StatefulPartitionedCall!conv3d_11/StatefulPartitionedCall2F
!conv3d_12/StatefulPartitionedCall!conv3d_12/StatefulPartitionedCall2F
!conv3d_13/StatefulPartitionedCall!conv3d_13/StatefulPartitionedCall2F
!conv3d_14/StatefulPartitionedCall!conv3d_14/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2R
'ensemble_output/StatefulPartitionedCall'ensemble_output/StatefulPartitionedCall:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs:]Y
5
_output_shapes#
!:???????????
 
_user_specified_nameinputs
??
?/
"__inference__wrapped_model_9333325

svm_inputs
img3d_inputsQ
3ensemble4d_conv3d_10_conv3d_readvariableop_resource: B
4ensemble4d_conv3d_10_biasadd_readvariableop_resource: G
9ensemble4d_batch_normalization_10_readvariableop_resource: I
;ensemble4d_batch_normalization_10_readvariableop_1_resource: X
Jensemble4d_batch_normalization_10_fusedbatchnormv3_readvariableop_resource: Z
Lensemble4d_batch_normalization_10_fusedbatchnormv3_readvariableop_1_resource: Q
3ensemble4d_conv3d_11_conv3d_readvariableop_resource:  B
4ensemble4d_conv3d_11_biasadd_readvariableop_resource: G
9ensemble4d_batch_normalization_11_readvariableop_resource: I
;ensemble4d_batch_normalization_11_readvariableop_1_resource: X
Jensemble4d_batch_normalization_11_fusedbatchnormv3_readvariableop_resource: Z
Lensemble4d_batch_normalization_11_fusedbatchnormv3_readvariableop_1_resource: Q
3ensemble4d_conv3d_12_conv3d_readvariableop_resource: @B
4ensemble4d_conv3d_12_biasadd_readvariableop_resource:@G
9ensemble4d_batch_normalization_12_readvariableop_resource:@I
;ensemble4d_batch_normalization_12_readvariableop_1_resource:@X
Jensemble4d_batch_normalization_12_fusedbatchnormv3_readvariableop_resource:@Z
Lensemble4d_batch_normalization_12_fusedbatchnormv3_readvariableop_1_resource:@R
3ensemble4d_conv3d_13_conv3d_readvariableop_resource:@?C
4ensemble4d_conv3d_13_biasadd_readvariableop_resource:	?H
9ensemble4d_batch_normalization_13_readvariableop_resource:	?J
;ensemble4d_batch_normalization_13_readvariableop_1_resource:	?Y
Jensemble4d_batch_normalization_13_fusedbatchnormv3_readvariableop_resource:	?[
Lensemble4d_batch_normalization_13_fusedbatchnormv3_readvariableop_1_resource:	?S
3ensemble4d_conv3d_14_conv3d_readvariableop_resource:??C
4ensemble4d_conv3d_14_biasadd_readvariableop_resource:	?E
3ensemble4d_1_dense18_matmul_readvariableop_resource:
B
4ensemble4d_1_dense18_biasadd_readvariableop_resource:H
9ensemble4d_batch_normalization_14_readvariableop_resource:	?J
;ensemble4d_batch_normalization_14_readvariableop_1_resource:	?Y
Jensemble4d_batch_normalization_14_fusedbatchnormv3_readvariableop_resource:	?[
Lensemble4d_batch_normalization_14_fusedbatchnormv3_readvariableop_1_resource:	?E
3ensemble4d_2_dense32_matmul_readvariableop_resource: B
4ensemble4d_2_dense32_biasadd_readvariableop_resource: E
3ensemble4d_3_dense64_matmul_readvariableop_resource: @B
4ensemble4d_3_dense64_biasadd_readvariableop_resource:@E
1ensemble4d_dense_2_matmul_readvariableop_resource:
??A
2ensemble4d_dense_2_biasadd_readvariableop_resource:	?E
3ensemble4d_4_dense32_matmul_readvariableop_resource:@ B
4ensemble4d_4_dense32_biasadd_readvariableop_resource: E
3ensemble4d_5_dense18_matmul_readvariableop_resource: B
4ensemble4d_5_dense18_biasadd_readvariableop_resource:K
8ensemble4d_combined_input_matmul_readvariableop_resource:	?	G
9ensemble4d_combined_input_biasadd_readvariableop_resource:	K
9ensemble4d_ensemble_output_matmul_readvariableop_resource:	H
:ensemble4d_ensemble_output_biasadd_readvariableop_resource:
identity??+ensemble4d/1_dense18/BiasAdd/ReadVariableOp?*ensemble4d/1_dense18/MatMul/ReadVariableOp?+ensemble4d/2_dense32/BiasAdd/ReadVariableOp?*ensemble4d/2_dense32/MatMul/ReadVariableOp?+ensemble4d/3_dense64/BiasAdd/ReadVariableOp?*ensemble4d/3_dense64/MatMul/ReadVariableOp?+ensemble4d/4_dense32/BiasAdd/ReadVariableOp?*ensemble4d/4_dense32/MatMul/ReadVariableOp?+ensemble4d/5_dense18/BiasAdd/ReadVariableOp?*ensemble4d/5_dense18/MatMul/ReadVariableOp?Aensemble4d/batch_normalization_10/FusedBatchNormV3/ReadVariableOp?Censemble4d/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1?0ensemble4d/batch_normalization_10/ReadVariableOp?2ensemble4d/batch_normalization_10/ReadVariableOp_1?Aensemble4d/batch_normalization_11/FusedBatchNormV3/ReadVariableOp?Censemble4d/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1?0ensemble4d/batch_normalization_11/ReadVariableOp?2ensemble4d/batch_normalization_11/ReadVariableOp_1?Aensemble4d/batch_normalization_12/FusedBatchNormV3/ReadVariableOp?Censemble4d/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1?0ensemble4d/batch_normalization_12/ReadVariableOp?2ensemble4d/batch_normalization_12/ReadVariableOp_1?Aensemble4d/batch_normalization_13/FusedBatchNormV3/ReadVariableOp?Censemble4d/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1?0ensemble4d/batch_normalization_13/ReadVariableOp?2ensemble4d/batch_normalization_13/ReadVariableOp_1?Aensemble4d/batch_normalization_14/FusedBatchNormV3/ReadVariableOp?Censemble4d/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1?0ensemble4d/batch_normalization_14/ReadVariableOp?2ensemble4d/batch_normalization_14/ReadVariableOp_1?0ensemble4d/combined_input/BiasAdd/ReadVariableOp?/ensemble4d/combined_input/MatMul/ReadVariableOp?+ensemble4d/conv3d_10/BiasAdd/ReadVariableOp?*ensemble4d/conv3d_10/Conv3D/ReadVariableOp?+ensemble4d/conv3d_11/BiasAdd/ReadVariableOp?*ensemble4d/conv3d_11/Conv3D/ReadVariableOp?+ensemble4d/conv3d_12/BiasAdd/ReadVariableOp?*ensemble4d/conv3d_12/Conv3D/ReadVariableOp?+ensemble4d/conv3d_13/BiasAdd/ReadVariableOp?*ensemble4d/conv3d_13/Conv3D/ReadVariableOp?+ensemble4d/conv3d_14/BiasAdd/ReadVariableOp?*ensemble4d/conv3d_14/Conv3D/ReadVariableOp?)ensemble4d/dense_2/BiasAdd/ReadVariableOp?(ensemble4d/dense_2/MatMul/ReadVariableOp?1ensemble4d/ensemble_output/BiasAdd/ReadVariableOp?0ensemble4d/ensemble_output/MatMul/ReadVariableOp?
*ensemble4d/conv3d_10/Conv3D/ReadVariableOpReadVariableOp3ensemble4d_conv3d_10_conv3d_readvariableop_resource**
_output_shapes
: *
dtype02,
*ensemble4d/conv3d_10/Conv3D/ReadVariableOp?
ensemble4d/conv3d_10/Conv3DConv3Dimg3d_inputs2ensemble4d/conv3d_10/Conv3D/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:??????????? *
paddingSAME*
strides	
2
ensemble4d/conv3d_10/Conv3D?
+ensemble4d/conv3d_10/BiasAdd/ReadVariableOpReadVariableOp4ensemble4d_conv3d_10_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+ensemble4d/conv3d_10/BiasAdd/ReadVariableOp?
ensemble4d/conv3d_10/BiasAddBiasAdd$ensemble4d/conv3d_10/Conv3D:output:03ensemble4d/conv3d_10/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:??????????? 2
ensemble4d/conv3d_10/BiasAdd?
ensemble4d/conv3d_10/LeakyRelu	LeakyRelu%ensemble4d/conv3d_10/BiasAdd:output:0*5
_output_shapes#
!:??????????? 2 
ensemble4d/conv3d_10/LeakyRelu?
%ensemble4d/max_pooling3d_10/MaxPool3D	MaxPool3D,ensemble4d/conv3d_10/LeakyRelu:activations:0*
T0*3
_output_shapes!
:?????????@@ *
ksize	
*
paddingVALID*
strides	
2'
%ensemble4d/max_pooling3d_10/MaxPool3D?
0ensemble4d/batch_normalization_10/ReadVariableOpReadVariableOp9ensemble4d_batch_normalization_10_readvariableop_resource*
_output_shapes
: *
dtype022
0ensemble4d/batch_normalization_10/ReadVariableOp?
2ensemble4d/batch_normalization_10/ReadVariableOp_1ReadVariableOp;ensemble4d_batch_normalization_10_readvariableop_1_resource*
_output_shapes
: *
dtype024
2ensemble4d/batch_normalization_10/ReadVariableOp_1?
Aensemble4d/batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOpJensemble4d_batch_normalization_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02C
Aensemble4d/batch_normalization_10/FusedBatchNormV3/ReadVariableOp?
Censemble4d/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLensemble4d_batch_normalization_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02E
Censemble4d/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1?
2ensemble4d/batch_normalization_10/FusedBatchNormV3FusedBatchNormV3.ensemble4d/max_pooling3d_10/MaxPool3D:output:08ensemble4d/batch_normalization_10/ReadVariableOp:value:0:ensemble4d/batch_normalization_10/ReadVariableOp_1:value:0Iensemble4d/batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0Kensemble4d/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*O
_output_shapes=
;:?????????@@ : : : : :*
data_formatNDHWC*
epsilon%o?:*
is_training( 24
2ensemble4d/batch_normalization_10/FusedBatchNormV3?
*ensemble4d/conv3d_11/Conv3D/ReadVariableOpReadVariableOp3ensemble4d_conv3d_11_conv3d_readvariableop_resource**
_output_shapes
:  *
dtype02,
*ensemble4d/conv3d_11/Conv3D/ReadVariableOp?
ensemble4d/conv3d_11/Conv3DConv3D6ensemble4d/batch_normalization_10/FusedBatchNormV3:y:02ensemble4d/conv3d_11/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@@ *
paddingSAME*
strides	
2
ensemble4d/conv3d_11/Conv3D?
+ensemble4d/conv3d_11/BiasAdd/ReadVariableOpReadVariableOp4ensemble4d_conv3d_11_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+ensemble4d/conv3d_11/BiasAdd/ReadVariableOp?
ensemble4d/conv3d_11/BiasAddBiasAdd$ensemble4d/conv3d_11/Conv3D:output:03ensemble4d/conv3d_11/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@@ 2
ensemble4d/conv3d_11/BiasAdd?
ensemble4d/conv3d_11/LeakyRelu	LeakyRelu%ensemble4d/conv3d_11/BiasAdd:output:0*3
_output_shapes!
:?????????@@ 2 
ensemble4d/conv3d_11/LeakyRelu?
%ensemble4d/max_pooling3d_11/MaxPool3D	MaxPool3D,ensemble4d/conv3d_11/LeakyRelu:activations:0*
T0*3
_output_shapes!
:?????????@@ *
ksize	
*
paddingVALID*
strides	
2'
%ensemble4d/max_pooling3d_11/MaxPool3D?
0ensemble4d/batch_normalization_11/ReadVariableOpReadVariableOp9ensemble4d_batch_normalization_11_readvariableop_resource*
_output_shapes
: *
dtype022
0ensemble4d/batch_normalization_11/ReadVariableOp?
2ensemble4d/batch_normalization_11/ReadVariableOp_1ReadVariableOp;ensemble4d_batch_normalization_11_readvariableop_1_resource*
_output_shapes
: *
dtype024
2ensemble4d/batch_normalization_11/ReadVariableOp_1?
Aensemble4d/batch_normalization_11/FusedBatchNormV3/ReadVariableOpReadVariableOpJensemble4d_batch_normalization_11_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02C
Aensemble4d/batch_normalization_11/FusedBatchNormV3/ReadVariableOp?
Censemble4d/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLensemble4d_batch_normalization_11_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02E
Censemble4d/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1?
2ensemble4d/batch_normalization_11/FusedBatchNormV3FusedBatchNormV3.ensemble4d/max_pooling3d_11/MaxPool3D:output:08ensemble4d/batch_normalization_11/ReadVariableOp:value:0:ensemble4d/batch_normalization_11/ReadVariableOp_1:value:0Iensemble4d/batch_normalization_11/FusedBatchNormV3/ReadVariableOp:value:0Kensemble4d/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*O
_output_shapes=
;:?????????@@ : : : : :*
data_formatNDHWC*
epsilon%o?:*
is_training( 24
2ensemble4d/batch_normalization_11/FusedBatchNormV3?
*ensemble4d/conv3d_12/Conv3D/ReadVariableOpReadVariableOp3ensemble4d_conv3d_12_conv3d_readvariableop_resource**
_output_shapes
: @*
dtype02,
*ensemble4d/conv3d_12/Conv3D/ReadVariableOp?
ensemble4d/conv3d_12/Conv3DConv3D6ensemble4d/batch_normalization_11/FusedBatchNormV3:y:02ensemble4d/conv3d_12/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@@@*
paddingSAME*
strides	
2
ensemble4d/conv3d_12/Conv3D?
+ensemble4d/conv3d_12/BiasAdd/ReadVariableOpReadVariableOp4ensemble4d_conv3d_12_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+ensemble4d/conv3d_12/BiasAdd/ReadVariableOp?
ensemble4d/conv3d_12/BiasAddBiasAdd$ensemble4d/conv3d_12/Conv3D:output:03ensemble4d/conv3d_12/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@@@2
ensemble4d/conv3d_12/BiasAdd?
ensemble4d/conv3d_12/LeakyRelu	LeakyRelu%ensemble4d/conv3d_12/BiasAdd:output:0*3
_output_shapes!
:?????????@@@2 
ensemble4d/conv3d_12/LeakyRelu?
%ensemble4d/max_pooling3d_12/MaxPool3D	MaxPool3D,ensemble4d/conv3d_12/LeakyRelu:activations:0*
T0*3
_output_shapes!
:?????????@@@*
ksize	
*
paddingVALID*
strides	
2'
%ensemble4d/max_pooling3d_12/MaxPool3D?
0ensemble4d/batch_normalization_12/ReadVariableOpReadVariableOp9ensemble4d_batch_normalization_12_readvariableop_resource*
_output_shapes
:@*
dtype022
0ensemble4d/batch_normalization_12/ReadVariableOp?
2ensemble4d/batch_normalization_12/ReadVariableOp_1ReadVariableOp;ensemble4d_batch_normalization_12_readvariableop_1_resource*
_output_shapes
:@*
dtype024
2ensemble4d/batch_normalization_12/ReadVariableOp_1?
Aensemble4d/batch_normalization_12/FusedBatchNormV3/ReadVariableOpReadVariableOpJensemble4d_batch_normalization_12_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02C
Aensemble4d/batch_normalization_12/FusedBatchNormV3/ReadVariableOp?
Censemble4d/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLensemble4d_batch_normalization_12_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02E
Censemble4d/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1?
2ensemble4d/batch_normalization_12/FusedBatchNormV3FusedBatchNormV3.ensemble4d/max_pooling3d_12/MaxPool3D:output:08ensemble4d/batch_normalization_12/ReadVariableOp:value:0:ensemble4d/batch_normalization_12/ReadVariableOp_1:value:0Iensemble4d/batch_normalization_12/FusedBatchNormV3/ReadVariableOp:value:0Kensemble4d/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*O
_output_shapes=
;:?????????@@@:@:@:@:@:*
data_formatNDHWC*
epsilon%o?:*
is_training( 24
2ensemble4d/batch_normalization_12/FusedBatchNormV3?
*ensemble4d/conv3d_13/Conv3D/ReadVariableOpReadVariableOp3ensemble4d_conv3d_13_conv3d_readvariableop_resource*+
_output_shapes
:@?*
dtype02,
*ensemble4d/conv3d_13/Conv3D/ReadVariableOp?
ensemble4d/conv3d_13/Conv3DConv3D6ensemble4d/batch_normalization_12/FusedBatchNormV3:y:02ensemble4d/conv3d_13/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :?????????@@?*
paddingSAME*
strides	
2
ensemble4d/conv3d_13/Conv3D?
+ensemble4d/conv3d_13/BiasAdd/ReadVariableOpReadVariableOp4ensemble4d_conv3d_13_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+ensemble4d/conv3d_13/BiasAdd/ReadVariableOp?
ensemble4d/conv3d_13/BiasAddBiasAdd$ensemble4d/conv3d_13/Conv3D:output:03ensemble4d/conv3d_13/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :?????????@@?2
ensemble4d/conv3d_13/BiasAdd?
ensemble4d/conv3d_13/LeakyRelu	LeakyRelu%ensemble4d/conv3d_13/BiasAdd:output:0*4
_output_shapes"
 :?????????@@?2 
ensemble4d/conv3d_13/LeakyRelu?
%ensemble4d/max_pooling3d_13/MaxPool3D	MaxPool3D,ensemble4d/conv3d_13/LeakyRelu:activations:0*
T0*4
_output_shapes"
 :?????????@@?*
ksize	
*
paddingVALID*
strides	
2'
%ensemble4d/max_pooling3d_13/MaxPool3D?
0ensemble4d/batch_normalization_13/ReadVariableOpReadVariableOp9ensemble4d_batch_normalization_13_readvariableop_resource*
_output_shapes	
:?*
dtype022
0ensemble4d/batch_normalization_13/ReadVariableOp?
2ensemble4d/batch_normalization_13/ReadVariableOp_1ReadVariableOp;ensemble4d_batch_normalization_13_readvariableop_1_resource*
_output_shapes	
:?*
dtype024
2ensemble4d/batch_normalization_13/ReadVariableOp_1?
Aensemble4d/batch_normalization_13/FusedBatchNormV3/ReadVariableOpReadVariableOpJensemble4d_batch_normalization_13_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02C
Aensemble4d/batch_normalization_13/FusedBatchNormV3/ReadVariableOp?
Censemble4d/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLensemble4d_batch_normalization_13_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02E
Censemble4d/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1?
2ensemble4d/batch_normalization_13/FusedBatchNormV3FusedBatchNormV3.ensemble4d/max_pooling3d_13/MaxPool3D:output:08ensemble4d/batch_normalization_13/ReadVariableOp:value:0:ensemble4d/batch_normalization_13/ReadVariableOp_1:value:0Iensemble4d/batch_normalization_13/FusedBatchNormV3/ReadVariableOp:value:0Kensemble4d/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*T
_output_shapesB
@:?????????@@?:?:?:?:?:*
data_formatNDHWC*
epsilon%o?:*
is_training( 24
2ensemble4d/batch_normalization_13/FusedBatchNormV3?
*ensemble4d/conv3d_14/Conv3D/ReadVariableOpReadVariableOp3ensemble4d_conv3d_14_conv3d_readvariableop_resource*,
_output_shapes
:??*
dtype02,
*ensemble4d/conv3d_14/Conv3D/ReadVariableOp?
ensemble4d/conv3d_14/Conv3DConv3D6ensemble4d/batch_normalization_13/FusedBatchNormV3:y:02ensemble4d/conv3d_14/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :?????????@@?*
paddingSAME*
strides	
2
ensemble4d/conv3d_14/Conv3D?
+ensemble4d/conv3d_14/BiasAdd/ReadVariableOpReadVariableOp4ensemble4d_conv3d_14_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+ensemble4d/conv3d_14/BiasAdd/ReadVariableOp?
ensemble4d/conv3d_14/BiasAddBiasAdd$ensemble4d/conv3d_14/Conv3D:output:03ensemble4d/conv3d_14/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :?????????@@?2
ensemble4d/conv3d_14/BiasAdd?
ensemble4d/conv3d_14/LeakyRelu	LeakyRelu%ensemble4d/conv3d_14/BiasAdd:output:0*4
_output_shapes"
 :?????????@@?2 
ensemble4d/conv3d_14/LeakyRelu?
%ensemble4d/max_pooling3d_14/MaxPool3D	MaxPool3D,ensemble4d/conv3d_14/LeakyRelu:activations:0*
T0*4
_output_shapes"
 :?????????@@?*
ksize	
*
paddingVALID*
strides	
2'
%ensemble4d/max_pooling3d_14/MaxPool3D?
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
0ensemble4d/batch_normalization_14/ReadVariableOpReadVariableOp9ensemble4d_batch_normalization_14_readvariableop_resource*
_output_shapes	
:?*
dtype022
0ensemble4d/batch_normalization_14/ReadVariableOp?
2ensemble4d/batch_normalization_14/ReadVariableOp_1ReadVariableOp;ensemble4d_batch_normalization_14_readvariableop_1_resource*
_output_shapes	
:?*
dtype024
2ensemble4d/batch_normalization_14/ReadVariableOp_1?
Aensemble4d/batch_normalization_14/FusedBatchNormV3/ReadVariableOpReadVariableOpJensemble4d_batch_normalization_14_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02C
Aensemble4d/batch_normalization_14/FusedBatchNormV3/ReadVariableOp?
Censemble4d/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLensemble4d_batch_normalization_14_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02E
Censemble4d/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1?
2ensemble4d/batch_normalization_14/FusedBatchNormV3FusedBatchNormV3.ensemble4d/max_pooling3d_14/MaxPool3D:output:08ensemble4d/batch_normalization_14/ReadVariableOp:value:0:ensemble4d/batch_normalization_14/ReadVariableOp_1:value:0Iensemble4d/batch_normalization_14/FusedBatchNormV3/ReadVariableOp:value:0Kensemble4d/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*T
_output_shapesB
@:?????????@@?:?:?:?:?:*
data_formatNDHWC*
epsilon%o?:*
is_training( 24
2ensemble4d/batch_normalization_14/FusedBatchNormV3?
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
<ensemble4d/global_average_pooling3d_2/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2>
<ensemble4d/global_average_pooling3d_2/Mean/reduction_indices?
*ensemble4d/global_average_pooling3d_2/MeanMean6ensemble4d/batch_normalization_14/FusedBatchNormV3:y:0Eensemble4d/global_average_pooling3d_2/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2,
*ensemble4d/global_average_pooling3d_2/Mean?
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
(ensemble4d/dense_2/MatMul/ReadVariableOpReadVariableOp1ensemble4d_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02*
(ensemble4d/dense_2/MatMul/ReadVariableOp?
ensemble4d/dense_2/MatMulMatMul3ensemble4d/global_average_pooling3d_2/Mean:output:00ensemble4d/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
ensemble4d/dense_2/MatMul?
)ensemble4d/dense_2/BiasAdd/ReadVariableOpReadVariableOp2ensemble4d_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)ensemble4d/dense_2/BiasAdd/ReadVariableOp?
ensemble4d/dense_2/BiasAddBiasAdd#ensemble4d/dense_2/MatMul:product:01ensemble4d/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
ensemble4d/dense_2/BiasAdd?
ensemble4d/dense_2/LeakyRelu	LeakyRelu#ensemble4d/dense_2/BiasAdd:output:0*(
_output_shapes
:??????????2
ensemble4d/dense_2/LeakyRelu?
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
ensemble4d/dropout_2/IdentityIdentity*ensemble4d/dense_2/LeakyRelu:activations:0*
T0*(
_output_shapes
:??????????2
ensemble4d/dropout_2/Identity?
$ensemble4d/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2&
$ensemble4d/concatenate_1/concat/axis?
ensemble4d/concatenate_1/concatConcatV2,ensemble4d/5_dense18/LeakyRelu:activations:0&ensemble4d/dropout_2/Identity:output:0-ensemble4d/concatenate_1/concat/axis:output:0*
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
NoOpNoOp,^ensemble4d/1_dense18/BiasAdd/ReadVariableOp+^ensemble4d/1_dense18/MatMul/ReadVariableOp,^ensemble4d/2_dense32/BiasAdd/ReadVariableOp+^ensemble4d/2_dense32/MatMul/ReadVariableOp,^ensemble4d/3_dense64/BiasAdd/ReadVariableOp+^ensemble4d/3_dense64/MatMul/ReadVariableOp,^ensemble4d/4_dense32/BiasAdd/ReadVariableOp+^ensemble4d/4_dense32/MatMul/ReadVariableOp,^ensemble4d/5_dense18/BiasAdd/ReadVariableOp+^ensemble4d/5_dense18/MatMul/ReadVariableOpB^ensemble4d/batch_normalization_10/FusedBatchNormV3/ReadVariableOpD^ensemble4d/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_11^ensemble4d/batch_normalization_10/ReadVariableOp3^ensemble4d/batch_normalization_10/ReadVariableOp_1B^ensemble4d/batch_normalization_11/FusedBatchNormV3/ReadVariableOpD^ensemble4d/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_11^ensemble4d/batch_normalization_11/ReadVariableOp3^ensemble4d/batch_normalization_11/ReadVariableOp_1B^ensemble4d/batch_normalization_12/FusedBatchNormV3/ReadVariableOpD^ensemble4d/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_11^ensemble4d/batch_normalization_12/ReadVariableOp3^ensemble4d/batch_normalization_12/ReadVariableOp_1B^ensemble4d/batch_normalization_13/FusedBatchNormV3/ReadVariableOpD^ensemble4d/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_11^ensemble4d/batch_normalization_13/ReadVariableOp3^ensemble4d/batch_normalization_13/ReadVariableOp_1B^ensemble4d/batch_normalization_14/FusedBatchNormV3/ReadVariableOpD^ensemble4d/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_11^ensemble4d/batch_normalization_14/ReadVariableOp3^ensemble4d/batch_normalization_14/ReadVariableOp_11^ensemble4d/combined_input/BiasAdd/ReadVariableOp0^ensemble4d/combined_input/MatMul/ReadVariableOp,^ensemble4d/conv3d_10/BiasAdd/ReadVariableOp+^ensemble4d/conv3d_10/Conv3D/ReadVariableOp,^ensemble4d/conv3d_11/BiasAdd/ReadVariableOp+^ensemble4d/conv3d_11/Conv3D/ReadVariableOp,^ensemble4d/conv3d_12/BiasAdd/ReadVariableOp+^ensemble4d/conv3d_12/Conv3D/ReadVariableOp,^ensemble4d/conv3d_13/BiasAdd/ReadVariableOp+^ensemble4d/conv3d_13/Conv3D/ReadVariableOp,^ensemble4d/conv3d_14/BiasAdd/ReadVariableOp+^ensemble4d/conv3d_14/Conv3D/ReadVariableOp*^ensemble4d/dense_2/BiasAdd/ReadVariableOp)^ensemble4d/dense_2/MatMul/ReadVariableOp2^ensemble4d/ensemble_output/BiasAdd/ReadVariableOp1^ensemble4d/ensemble_output/MatMul/ReadVariableOp*"
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
Aensemble4d/batch_normalization_10/FusedBatchNormV3/ReadVariableOpAensemble4d/batch_normalization_10/FusedBatchNormV3/ReadVariableOp2?
Censemble4d/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1Censemble4d/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_12d
0ensemble4d/batch_normalization_10/ReadVariableOp0ensemble4d/batch_normalization_10/ReadVariableOp2h
2ensemble4d/batch_normalization_10/ReadVariableOp_12ensemble4d/batch_normalization_10/ReadVariableOp_12?
Aensemble4d/batch_normalization_11/FusedBatchNormV3/ReadVariableOpAensemble4d/batch_normalization_11/FusedBatchNormV3/ReadVariableOp2?
Censemble4d/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1Censemble4d/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_12d
0ensemble4d/batch_normalization_11/ReadVariableOp0ensemble4d/batch_normalization_11/ReadVariableOp2h
2ensemble4d/batch_normalization_11/ReadVariableOp_12ensemble4d/batch_normalization_11/ReadVariableOp_12?
Aensemble4d/batch_normalization_12/FusedBatchNormV3/ReadVariableOpAensemble4d/batch_normalization_12/FusedBatchNormV3/ReadVariableOp2?
Censemble4d/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1Censemble4d/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_12d
0ensemble4d/batch_normalization_12/ReadVariableOp0ensemble4d/batch_normalization_12/ReadVariableOp2h
2ensemble4d/batch_normalization_12/ReadVariableOp_12ensemble4d/batch_normalization_12/ReadVariableOp_12?
Aensemble4d/batch_normalization_13/FusedBatchNormV3/ReadVariableOpAensemble4d/batch_normalization_13/FusedBatchNormV3/ReadVariableOp2?
Censemble4d/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1Censemble4d/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_12d
0ensemble4d/batch_normalization_13/ReadVariableOp0ensemble4d/batch_normalization_13/ReadVariableOp2h
2ensemble4d/batch_normalization_13/ReadVariableOp_12ensemble4d/batch_normalization_13/ReadVariableOp_12?
Aensemble4d/batch_normalization_14/FusedBatchNormV3/ReadVariableOpAensemble4d/batch_normalization_14/FusedBatchNormV3/ReadVariableOp2?
Censemble4d/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1Censemble4d/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_12d
0ensemble4d/batch_normalization_14/ReadVariableOp0ensemble4d/batch_normalization_14/ReadVariableOp2h
2ensemble4d/batch_normalization_14/ReadVariableOp_12ensemble4d/batch_normalization_14/ReadVariableOp_12d
0ensemble4d/combined_input/BiasAdd/ReadVariableOp0ensemble4d/combined_input/BiasAdd/ReadVariableOp2b
/ensemble4d/combined_input/MatMul/ReadVariableOp/ensemble4d/combined_input/MatMul/ReadVariableOp2Z
+ensemble4d/conv3d_10/BiasAdd/ReadVariableOp+ensemble4d/conv3d_10/BiasAdd/ReadVariableOp2X
*ensemble4d/conv3d_10/Conv3D/ReadVariableOp*ensemble4d/conv3d_10/Conv3D/ReadVariableOp2Z
+ensemble4d/conv3d_11/BiasAdd/ReadVariableOp+ensemble4d/conv3d_11/BiasAdd/ReadVariableOp2X
*ensemble4d/conv3d_11/Conv3D/ReadVariableOp*ensemble4d/conv3d_11/Conv3D/ReadVariableOp2Z
+ensemble4d/conv3d_12/BiasAdd/ReadVariableOp+ensemble4d/conv3d_12/BiasAdd/ReadVariableOp2X
*ensemble4d/conv3d_12/Conv3D/ReadVariableOp*ensemble4d/conv3d_12/Conv3D/ReadVariableOp2Z
+ensemble4d/conv3d_13/BiasAdd/ReadVariableOp+ensemble4d/conv3d_13/BiasAdd/ReadVariableOp2X
*ensemble4d/conv3d_13/Conv3D/ReadVariableOp*ensemble4d/conv3d_13/Conv3D/ReadVariableOp2Z
+ensemble4d/conv3d_14/BiasAdd/ReadVariableOp+ensemble4d/conv3d_14/BiasAdd/ReadVariableOp2X
*ensemble4d/conv3d_14/Conv3D/ReadVariableOp*ensemble4d/conv3d_14/Conv3D/ReadVariableOp2V
)ensemble4d/dense_2/BiasAdd/ReadVariableOp)ensemble4d/dense_2/BiasAdd/ReadVariableOp2T
(ensemble4d/dense_2/MatMul/ReadVariableOp(ensemble4d/dense_2/MatMul/ReadVariableOp2f
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
?
?
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_9336609

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
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_9334288

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
?
?
F__inference_conv3d_12_layer_call_and_return_conditional_losses_9336683

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
?
8__inference_batch_normalization_12_layer_call_fn_9336742

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
GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_93342382
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
?
i
M__inference_max_pooling3d_13_layer_call_and_return_conditional_losses_9333778

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
N
2__inference_max_pooling3d_14_layer_call_fn_9337041

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
GPU2*0J 8? *V
fQRO
M__inference_max_pooling3d_14_layer_call_and_return_conditional_losses_93343192
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
?
?
F__inference_4_dense32_layer_call_and_return_conditional_losses_9337257

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
i
M__inference_max_pooling3d_10_layer_call_and_return_conditional_losses_9333334

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
?
?
F__inference_5_dense18_layer_call_and_return_conditional_losses_9334451

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
?
d
+__inference_dropout_2_layer_call_fn_9337307

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
F__inference_dropout_2_layer_call_and_return_conditional_losses_93346502
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
?	
?
8__inference_batch_normalization_10_layer_call_fn_9336401

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
GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_93334132
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
?
8__inference_batch_normalization_10_layer_call_fn_9336414

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
GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_93341382
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
??
?S
#__inference__traced_restore_9338167
file_prefix?
!assignvariableop_conv3d_10_kernel: /
!assignvariableop_1_conv3d_10_bias: =
/assignvariableop_2_batch_normalization_10_gamma: <
.assignvariableop_3_batch_normalization_10_beta: C
5assignvariableop_4_batch_normalization_10_moving_mean: G
9assignvariableop_5_batch_normalization_10_moving_variance: A
#assignvariableop_6_conv3d_11_kernel:  /
!assignvariableop_7_conv3d_11_bias: =
/assignvariableop_8_batch_normalization_11_gamma: <
.assignvariableop_9_batch_normalization_11_beta: D
6assignvariableop_10_batch_normalization_11_moving_mean: H
:assignvariableop_11_batch_normalization_11_moving_variance: B
$assignvariableop_12_conv3d_12_kernel: @0
"assignvariableop_13_conv3d_12_bias:@>
0assignvariableop_14_batch_normalization_12_gamma:@=
/assignvariableop_15_batch_normalization_12_beta:@D
6assignvariableop_16_batch_normalization_12_moving_mean:@H
:assignvariableop_17_batch_normalization_12_moving_variance:@C
$assignvariableop_18_conv3d_13_kernel:@?1
"assignvariableop_19_conv3d_13_bias:	??
0assignvariableop_20_batch_normalization_13_gamma:	?>
/assignvariableop_21_batch_normalization_13_beta:	?E
6assignvariableop_22_batch_normalization_13_moving_mean:	?I
:assignvariableop_23_batch_normalization_13_moving_variance:	?D
$assignvariableop_24_conv3d_14_kernel:??1
"assignvariableop_25_conv3d_14_bias:	?6
$assignvariableop_26_1_dense18_kernel:
0
"assignvariableop_27_1_dense18_bias:6
$assignvariableop_28_2_dense32_kernel: 0
"assignvariableop_29_2_dense32_bias: ?
0assignvariableop_30_batch_normalization_14_gamma:	?>
/assignvariableop_31_batch_normalization_14_beta:	?E
6assignvariableop_32_batch_normalization_14_moving_mean:	?I
:assignvariableop_33_batch_normalization_14_moving_variance:	?6
$assignvariableop_34_3_dense64_kernel: @0
"assignvariableop_35_3_dense64_bias:@6
$assignvariableop_36_4_dense32_kernel:@ 0
"assignvariableop_37_4_dense32_bias: 6
"assignvariableop_38_dense_2_kernel:
??/
 assignvariableop_39_dense_2_bias:	?6
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
assignvariableop_53_count_1: I
+assignvariableop_54_adam_conv3d_10_kernel_m: 7
)assignvariableop_55_adam_conv3d_10_bias_m: E
7assignvariableop_56_adam_batch_normalization_10_gamma_m: D
6assignvariableop_57_adam_batch_normalization_10_beta_m: I
+assignvariableop_58_adam_conv3d_11_kernel_m:  7
)assignvariableop_59_adam_conv3d_11_bias_m: E
7assignvariableop_60_adam_batch_normalization_11_gamma_m: D
6assignvariableop_61_adam_batch_normalization_11_beta_m: I
+assignvariableop_62_adam_conv3d_12_kernel_m: @7
)assignvariableop_63_adam_conv3d_12_bias_m:@E
7assignvariableop_64_adam_batch_normalization_12_gamma_m:@D
6assignvariableop_65_adam_batch_normalization_12_beta_m:@J
+assignvariableop_66_adam_conv3d_13_kernel_m:@?8
)assignvariableop_67_adam_conv3d_13_bias_m:	?F
7assignvariableop_68_adam_batch_normalization_13_gamma_m:	?E
6assignvariableop_69_adam_batch_normalization_13_beta_m:	?K
+assignvariableop_70_adam_conv3d_14_kernel_m:??8
)assignvariableop_71_adam_conv3d_14_bias_m:	?=
+assignvariableop_72_adam_1_dense18_kernel_m:
7
)assignvariableop_73_adam_1_dense18_bias_m:=
+assignvariableop_74_adam_2_dense32_kernel_m: 7
)assignvariableop_75_adam_2_dense32_bias_m: F
7assignvariableop_76_adam_batch_normalization_14_gamma_m:	?E
6assignvariableop_77_adam_batch_normalization_14_beta_m:	?=
+assignvariableop_78_adam_3_dense64_kernel_m: @7
)assignvariableop_79_adam_3_dense64_bias_m:@=
+assignvariableop_80_adam_4_dense32_kernel_m:@ 7
)assignvariableop_81_adam_4_dense32_bias_m: =
)assignvariableop_82_adam_dense_2_kernel_m:
??6
'assignvariableop_83_adam_dense_2_bias_m:	?=
+assignvariableop_84_adam_5_dense18_kernel_m: 7
)assignvariableop_85_adam_5_dense18_bias_m:C
0assignvariableop_86_adam_combined_input_kernel_m:	?	<
.assignvariableop_87_adam_combined_input_bias_m:	C
1assignvariableop_88_adam_ensemble_output_kernel_m:	=
/assignvariableop_89_adam_ensemble_output_bias_m:I
+assignvariableop_90_adam_conv3d_10_kernel_v: 7
)assignvariableop_91_adam_conv3d_10_bias_v: E
7assignvariableop_92_adam_batch_normalization_10_gamma_v: D
6assignvariableop_93_adam_batch_normalization_10_beta_v: I
+assignvariableop_94_adam_conv3d_11_kernel_v:  7
)assignvariableop_95_adam_conv3d_11_bias_v: E
7assignvariableop_96_adam_batch_normalization_11_gamma_v: D
6assignvariableop_97_adam_batch_normalization_11_beta_v: I
+assignvariableop_98_adam_conv3d_12_kernel_v: @7
)assignvariableop_99_adam_conv3d_12_bias_v:@F
8assignvariableop_100_adam_batch_normalization_12_gamma_v:@E
7assignvariableop_101_adam_batch_normalization_12_beta_v:@K
,assignvariableop_102_adam_conv3d_13_kernel_v:@?9
*assignvariableop_103_adam_conv3d_13_bias_v:	?G
8assignvariableop_104_adam_batch_normalization_13_gamma_v:	?F
7assignvariableop_105_adam_batch_normalization_13_beta_v:	?L
,assignvariableop_106_adam_conv3d_14_kernel_v:??9
*assignvariableop_107_adam_conv3d_14_bias_v:	?>
,assignvariableop_108_adam_1_dense18_kernel_v:
8
*assignvariableop_109_adam_1_dense18_bias_v:>
,assignvariableop_110_adam_2_dense32_kernel_v: 8
*assignvariableop_111_adam_2_dense32_bias_v: G
8assignvariableop_112_adam_batch_normalization_14_gamma_v:	?F
7assignvariableop_113_adam_batch_normalization_14_beta_v:	?>
,assignvariableop_114_adam_3_dense64_kernel_v: @8
*assignvariableop_115_adam_3_dense64_bias_v:@>
,assignvariableop_116_adam_4_dense32_kernel_v:@ 8
*assignvariableop_117_adam_4_dense32_bias_v: >
*assignvariableop_118_adam_dense_2_kernel_v:
??7
(assignvariableop_119_adam_dense_2_bias_v:	?>
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
AssignVariableOpAssignVariableOp!assignvariableop_conv3d_10_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv3d_10_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp/assignvariableop_2_batch_normalization_10_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp.assignvariableop_3_batch_normalization_10_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp5assignvariableop_4_batch_normalization_10_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp9assignvariableop_5_batch_normalization_10_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp#assignvariableop_6_conv3d_11_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp!assignvariableop_7_conv3d_11_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp/assignvariableop_8_batch_normalization_11_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp.assignvariableop_9_batch_normalization_11_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp6assignvariableop_10_batch_normalization_11_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp:assignvariableop_11_batch_normalization_11_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp$assignvariableop_12_conv3d_12_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp"assignvariableop_13_conv3d_12_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp0assignvariableop_14_batch_normalization_12_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp/assignvariableop_15_batch_normalization_12_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp6assignvariableop_16_batch_normalization_12_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp:assignvariableop_17_batch_normalization_12_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp$assignvariableop_18_conv3d_13_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp"assignvariableop_19_conv3d_13_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp0assignvariableop_20_batch_normalization_13_gammaIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp/assignvariableop_21_batch_normalization_13_betaIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp6assignvariableop_22_batch_normalization_13_moving_meanIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp:assignvariableop_23_batch_normalization_13_moving_varianceIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp$assignvariableop_24_conv3d_14_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp"assignvariableop_25_conv3d_14_biasIdentity_25:output:0"/device:CPU:0*
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
AssignVariableOp_30AssignVariableOp0assignvariableop_30_batch_normalization_14_gammaIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp/assignvariableop_31_batch_normalization_14_betaIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp6assignvariableop_32_batch_normalization_14_moving_meanIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp:assignvariableop_33_batch_normalization_14_moving_varianceIdentity_33:output:0"/device:CPU:0*
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
AssignVariableOp_38AssignVariableOp"assignvariableop_38_dense_2_kernelIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp assignvariableop_39_dense_2_biasIdentity_39:output:0"/device:CPU:0*
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
AssignVariableOp_54AssignVariableOp+assignvariableop_54_adam_conv3d_10_kernel_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOp)assignvariableop_55_adam_conv3d_10_bias_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOp7assignvariableop_56_adam_batch_normalization_10_gamma_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOp6assignvariableop_57_adam_batch_normalization_10_beta_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOp+assignvariableop_58_adam_conv3d_11_kernel_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59?
AssignVariableOp_59AssignVariableOp)assignvariableop_59_adam_conv3d_11_bias_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60?
AssignVariableOp_60AssignVariableOp7assignvariableop_60_adam_batch_normalization_11_gamma_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61?
AssignVariableOp_61AssignVariableOp6assignvariableop_61_adam_batch_normalization_11_beta_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62?
AssignVariableOp_62AssignVariableOp+assignvariableop_62_adam_conv3d_12_kernel_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63?
AssignVariableOp_63AssignVariableOp)assignvariableop_63_adam_conv3d_12_bias_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64?
AssignVariableOp_64AssignVariableOp7assignvariableop_64_adam_batch_normalization_12_gamma_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65?
AssignVariableOp_65AssignVariableOp6assignvariableop_65_adam_batch_normalization_12_beta_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66?
AssignVariableOp_66AssignVariableOp+assignvariableop_66_adam_conv3d_13_kernel_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67?
AssignVariableOp_67AssignVariableOp)assignvariableop_67_adam_conv3d_13_bias_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68?
AssignVariableOp_68AssignVariableOp7assignvariableop_68_adam_batch_normalization_13_gamma_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69?
AssignVariableOp_69AssignVariableOp6assignvariableop_69_adam_batch_normalization_13_beta_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70?
AssignVariableOp_70AssignVariableOp+assignvariableop_70_adam_conv3d_14_kernel_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71?
AssignVariableOp_71AssignVariableOp)assignvariableop_71_adam_conv3d_14_bias_mIdentity_71:output:0"/device:CPU:0*
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
AssignVariableOp_76AssignVariableOp7assignvariableop_76_adam_batch_normalization_14_gamma_mIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77?
AssignVariableOp_77AssignVariableOp6assignvariableop_77_adam_batch_normalization_14_beta_mIdentity_77:output:0"/device:CPU:0*
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
AssignVariableOp_82AssignVariableOp)assignvariableop_82_adam_dense_2_kernel_mIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_82n
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:2
Identity_83?
AssignVariableOp_83AssignVariableOp'assignvariableop_83_adam_dense_2_bias_mIdentity_83:output:0"/device:CPU:0*
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
AssignVariableOp_90AssignVariableOp+assignvariableop_90_adam_conv3d_10_kernel_vIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_90n
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:2
Identity_91?
AssignVariableOp_91AssignVariableOp)assignvariableop_91_adam_conv3d_10_bias_vIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_91n
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:2
Identity_92?
AssignVariableOp_92AssignVariableOp7assignvariableop_92_adam_batch_normalization_10_gamma_vIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_92n
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:2
Identity_93?
AssignVariableOp_93AssignVariableOp6assignvariableop_93_adam_batch_normalization_10_beta_vIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_93n
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:2
Identity_94?
AssignVariableOp_94AssignVariableOp+assignvariableop_94_adam_conv3d_11_kernel_vIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_94n
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:2
Identity_95?
AssignVariableOp_95AssignVariableOp)assignvariableop_95_adam_conv3d_11_bias_vIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_95n
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:2
Identity_96?
AssignVariableOp_96AssignVariableOp7assignvariableop_96_adam_batch_normalization_11_gamma_vIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_96n
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:2
Identity_97?
AssignVariableOp_97AssignVariableOp6assignvariableop_97_adam_batch_normalization_11_beta_vIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_97n
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:2
Identity_98?
AssignVariableOp_98AssignVariableOp+assignvariableop_98_adam_conv3d_12_kernel_vIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_98n
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:2
Identity_99?
AssignVariableOp_99AssignVariableOp)assignvariableop_99_adam_conv3d_12_bias_vIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_99q
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:2
Identity_100?
AssignVariableOp_100AssignVariableOp8assignvariableop_100_adam_batch_normalization_12_gamma_vIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_100q
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:2
Identity_101?
AssignVariableOp_101AssignVariableOp7assignvariableop_101_adam_batch_normalization_12_beta_vIdentity_101:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_101q
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:2
Identity_102?
AssignVariableOp_102AssignVariableOp,assignvariableop_102_adam_conv3d_13_kernel_vIdentity_102:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_102q
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:2
Identity_103?
AssignVariableOp_103AssignVariableOp*assignvariableop_103_adam_conv3d_13_bias_vIdentity_103:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_103q
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:2
Identity_104?
AssignVariableOp_104AssignVariableOp8assignvariableop_104_adam_batch_normalization_13_gamma_vIdentity_104:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_104q
Identity_105IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:2
Identity_105?
AssignVariableOp_105AssignVariableOp7assignvariableop_105_adam_batch_normalization_13_beta_vIdentity_105:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_105q
Identity_106IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:2
Identity_106?
AssignVariableOp_106AssignVariableOp,assignvariableop_106_adam_conv3d_14_kernel_vIdentity_106:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_106q
Identity_107IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
:2
Identity_107?
AssignVariableOp_107AssignVariableOp*assignvariableop_107_adam_conv3d_14_bias_vIdentity_107:output:0"/device:CPU:0*
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
AssignVariableOp_112AssignVariableOp8assignvariableop_112_adam_batch_normalization_14_gamma_vIdentity_112:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_112q
Identity_113IdentityRestoreV2:tensors:113"/device:CPU:0*
T0*
_output_shapes
:2
Identity_113?
AssignVariableOp_113AssignVariableOp7assignvariableop_113_adam_batch_normalization_14_beta_vIdentity_113:output:0"/device:CPU:0*
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
AssignVariableOp_118AssignVariableOp*assignvariableop_118_adam_dense_2_kernel_vIdentity_118:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_118q
Identity_119IdentityRestoreV2:tensors:119"/device:CPU:0*
T0*
_output_shapes
:2
Identity_119?
AssignVariableOp_119AssignVariableOp(assignvariableop_119_adam_dense_2_bias_vIdentity_119:output:0"/device:CPU:0*
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
8__inference_batch_normalization_11_layer_call_fn_9336578

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
GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_93341882
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
?
?
,__inference_ensemble4d_layer_call_fn_9334603

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
G__inference_ensemble4d_layer_call_and_return_conditional_losses_93345082
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
?
i
M__inference_max_pooling3d_12_layer_call_and_return_conditional_losses_9336703

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
?
?
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_9336463

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
?
?
,__inference_ensemble4d_layer_call_fn_9335976
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
G__inference_ensemble4d_layer_call_and_return_conditional_losses_93352372
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
?
?
F__inference_conv3d_11_layer_call_and_return_conditional_losses_9334159

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
?
?
K__inference_combined_input_layer_call_and_return_conditional_losses_9337357

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
D__inference_dense_2_layer_call_and_return_conditional_losses_9337277

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
?
v
J__inference_concatenate_1_layer_call_and_return_conditional_losses_9337337
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
?
?
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_9336809

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
?
?
+__inference_conv3d_14_layer_call_fn_9337000

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
GPU2*0J 8? *O
fJRH
F__inference_conv3d_14_layer_call_and_return_conditional_losses_93343092
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
?
?
S__inference_batch_normalization_14_layer_call_and_return_conditional_losses_9337159

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
F__inference_5_dense18_layer_call_and_return_conditional_losses_9337297

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
?
?
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_9336937

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
?
?
)__inference_dense_2_layer_call_fn_9337266

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
D__inference_dense_2_layer_call_and_return_conditional_losses_93344172
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
?

?
8__inference_batch_normalization_13_layer_call_fn_9336880

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
GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_93338132
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
?
?
D__inference_dense_2_layer_call_and_return_conditional_losses_9334417

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
?
?
S__inference_batch_normalization_14_layer_call_and_return_conditional_losses_9333961

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
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
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
?__call__
+?&call_and_return_all_conditional_losses
?_default_save_signature"
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
 ?layer_regularization_losses
	variables
?non_trainable_variables
trainable_variables
?metrics
 regularization_losses
?layer_metrics
?layers
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
.:, 2conv3d_10/kernel
: 2conv3d_10/bias
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
 ?layer_regularization_losses
%	variables
?non_trainable_variables
&trainable_variables
?metrics
'regularization_losses
?layer_metrics
?layers
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
 ?layer_regularization_losses
)	variables
?non_trainable_variables
*trainable_variables
?metrics
+regularization_losses
?layer_metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:( 2batch_normalization_10/gamma
):' 2batch_normalization_10/beta
2:0  (2"batch_normalization_10/moving_mean
6:4  (2&batch_normalization_10/moving_variance
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
 ?layer_regularization_losses
2	variables
?non_trainable_variables
3trainable_variables
?metrics
4regularization_losses
?layer_metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.:,  2conv3d_11/kernel
: 2conv3d_11/bias
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
 ?layer_regularization_losses
8	variables
?non_trainable_variables
9trainable_variables
?metrics
:regularization_losses
?layer_metrics
?layers
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
 ?layer_regularization_losses
<	variables
?non_trainable_variables
=trainable_variables
?metrics
>regularization_losses
?layer_metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:( 2batch_normalization_11/gamma
):' 2batch_normalization_11/beta
2:0  (2"batch_normalization_11/moving_mean
6:4  (2&batch_normalization_11/moving_variance
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
 ?layer_regularization_losses
E	variables
?non_trainable_variables
Ftrainable_variables
?metrics
Gregularization_losses
?layer_metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.:, @2conv3d_12/kernel
:@2conv3d_12/bias
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
 ?layer_regularization_losses
K	variables
?non_trainable_variables
Ltrainable_variables
?metrics
Mregularization_losses
?layer_metrics
?layers
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
 ?layer_regularization_losses
O	variables
?non_trainable_variables
Ptrainable_variables
?metrics
Qregularization_losses
?layer_metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(@2batch_normalization_12/gamma
):'@2batch_normalization_12/beta
2:0@ (2"batch_normalization_12/moving_mean
6:4@ (2&batch_normalization_12/moving_variance
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
 ?layer_regularization_losses
X	variables
?non_trainable_variables
Ytrainable_variables
?metrics
Zregularization_losses
?layer_metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
/:-@?2conv3d_13/kernel
:?2conv3d_13/bias
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
 ?layer_regularization_losses
^	variables
?non_trainable_variables
_trainable_variables
?metrics
`regularization_losses
?layer_metrics
?layers
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
 ?layer_regularization_losses
b	variables
?non_trainable_variables
ctrainable_variables
?metrics
dregularization_losses
?layer_metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:)?2batch_normalization_13/gamma
*:(?2batch_normalization_13/beta
3:1? (2"batch_normalization_13/moving_mean
7:5? (2&batch_normalization_13/moving_variance
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
 ?layer_regularization_losses
k	variables
?non_trainable_variables
ltrainable_variables
?metrics
mregularization_losses
?layer_metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
0:.??2conv3d_14/kernel
:?2conv3d_14/bias
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
 ?layer_regularization_losses
q	variables
?non_trainable_variables
rtrainable_variables
?metrics
sregularization_losses
?layer_metrics
?layers
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
 ?layer_regularization_losses
w	variables
?non_trainable_variables
xtrainable_variables
?metrics
yregularization_losses
?layer_metrics
?layers
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
 ?layer_regularization_losses
{	variables
?non_trainable_variables
|trainable_variables
?metrics
}regularization_losses
?layer_metrics
?layers
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
 ?layer_regularization_losses
?	variables
?non_trainable_variables
?trainable_variables
?metrics
?regularization_losses
?layer_metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:)?2batch_normalization_14/gamma
*:(?2batch_normalization_14/beta
3:1? (2"batch_normalization_14/moving_mean
7:5? (2&batch_normalization_14/moving_variance
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
 ?layer_regularization_losses
?	variables
?non_trainable_variables
?trainable_variables
?metrics
?regularization_losses
?layer_metrics
?layers
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
 ?layer_regularization_losses
?	variables
?non_trainable_variables
?trainable_variables
?metrics
?regularization_losses
?layer_metrics
?layers
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
 ?layer_regularization_losses
?	variables
?non_trainable_variables
?trainable_variables
?metrics
?regularization_losses
?layer_metrics
?layers
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
 ?layer_regularization_losses
?	variables
?non_trainable_variables
?trainable_variables
?metrics
?regularization_losses
?layer_metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 
??2dense_2/kernel
:?2dense_2/bias
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
 ?layer_regularization_losses
?	variables
?non_trainable_variables
?trainable_variables
?metrics
?regularization_losses
?layer_metrics
?layers
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
 ?layer_regularization_losses
?	variables
?non_trainable_variables
?trainable_variables
?metrics
?regularization_losses
?layer_metrics
?layers
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
 ?layer_regularization_losses
?	variables
?non_trainable_variables
?trainable_variables
?metrics
?regularization_losses
?layer_metrics
?layers
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
 ?layer_regularization_losses
?	variables
?non_trainable_variables
?trainable_variables
?metrics
?regularization_losses
?layer_metrics
?layers
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
 ?layer_regularization_losses
?	variables
?non_trainable_variables
?trainable_variables
?metrics
?regularization_losses
?layer_metrics
?layers
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
 ?layer_regularization_losses
?	variables
?non_trainable_variables
?trainable_variables
?metrics
?regularization_losses
?layer_metrics
?layers
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
0
?0
?1"
trackable_list_wrapper
 "
trackable_dict_wrapper
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
.
00
11"
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
.
C0
D1"
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
.
V0
W1"
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
.
i0
j1"
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
0
?0
?1"
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
3:1 2Adam/conv3d_10/kernel/m
!: 2Adam/conv3d_10/bias/m
/:- 2#Adam/batch_normalization_10/gamma/m
.:, 2"Adam/batch_normalization_10/beta/m
3:1  2Adam/conv3d_11/kernel/m
!: 2Adam/conv3d_11/bias/m
/:- 2#Adam/batch_normalization_11/gamma/m
.:, 2"Adam/batch_normalization_11/beta/m
3:1 @2Adam/conv3d_12/kernel/m
!:@2Adam/conv3d_12/bias/m
/:-@2#Adam/batch_normalization_12/gamma/m
.:,@2"Adam/batch_normalization_12/beta/m
4:2@?2Adam/conv3d_13/kernel/m
": ?2Adam/conv3d_13/bias/m
0:.?2#Adam/batch_normalization_13/gamma/m
/:-?2"Adam/batch_normalization_13/beta/m
5:3??2Adam/conv3d_14/kernel/m
": ?2Adam/conv3d_14/bias/m
':%
2Adam/1_dense18/kernel/m
!:2Adam/1_dense18/bias/m
':% 2Adam/2_dense32/kernel/m
!: 2Adam/2_dense32/bias/m
0:.?2#Adam/batch_normalization_14/gamma/m
/:-?2"Adam/batch_normalization_14/beta/m
':% @2Adam/3_dense64/kernel/m
!:@2Adam/3_dense64/bias/m
':%@ 2Adam/4_dense32/kernel/m
!: 2Adam/4_dense32/bias/m
':%
??2Adam/dense_2/kernel/m
 :?2Adam/dense_2/bias/m
':% 2Adam/5_dense18/kernel/m
!:2Adam/5_dense18/bias/m
-:+	?	2Adam/combined_input/kernel/m
&:$	2Adam/combined_input/bias/m
-:+	2Adam/ensemble_output/kernel/m
':%2Adam/ensemble_output/bias/m
3:1 2Adam/conv3d_10/kernel/v
!: 2Adam/conv3d_10/bias/v
/:- 2#Adam/batch_normalization_10/gamma/v
.:, 2"Adam/batch_normalization_10/beta/v
3:1  2Adam/conv3d_11/kernel/v
!: 2Adam/conv3d_11/bias/v
/:- 2#Adam/batch_normalization_11/gamma/v
.:, 2"Adam/batch_normalization_11/beta/v
3:1 @2Adam/conv3d_12/kernel/v
!:@2Adam/conv3d_12/bias/v
/:-@2#Adam/batch_normalization_12/gamma/v
.:,@2"Adam/batch_normalization_12/beta/v
4:2@?2Adam/conv3d_13/kernel/v
": ?2Adam/conv3d_13/bias/v
0:.?2#Adam/batch_normalization_13/gamma/v
/:-?2"Adam/batch_normalization_13/beta/v
5:3??2Adam/conv3d_14/kernel/v
": ?2Adam/conv3d_14/bias/v
':%
2Adam/1_dense18/kernel/v
!:2Adam/1_dense18/bias/v
':% 2Adam/2_dense32/kernel/v
!: 2Adam/2_dense32/bias/v
0:.?2#Adam/batch_normalization_14/gamma/v
/:-?2"Adam/batch_normalization_14/beta/v
':% @2Adam/3_dense64/kernel/v
!:@2Adam/3_dense64/bias/v
':%@ 2Adam/4_dense32/kernel/v
!: 2Adam/4_dense32/bias/v
':%
??2Adam/dense_2/kernel/v
 :?2Adam/dense_2/bias/v
':% 2Adam/5_dense18/kernel/v
!:2Adam/5_dense18/bias/v
-:+	?	2Adam/combined_input/kernel/v
&:$	2Adam/combined_input/bias/v
-:+	2Adam/ensemble_output/kernel/v
':%2Adam/ensemble_output/bias/v
?2?
,__inference_ensemble4d_layer_call_fn_9334603
,__inference_ensemble4d_layer_call_fn_9335878
,__inference_ensemble4d_layer_call_fn_9335976
,__inference_ensemble4d_layer_call_fn_9335430?
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
G__inference_ensemble4d_layer_call_and_return_conditional_losses_9336152
G__inference_ensemble4d_layer_call_and_return_conditional_losses_9336335
G__inference_ensemble4d_layer_call_and_return_conditional_losses_9335553
G__inference_ensemble4d_layer_call_and_return_conditional_losses_9335676?
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
"__inference__wrapped_model_9333325
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
?2?
+__inference_conv3d_10_layer_call_fn_9336344?
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
F__inference_conv3d_10_layer_call_and_return_conditional_losses_9336355?
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
2__inference_max_pooling3d_10_layer_call_fn_9336360
2__inference_max_pooling3d_10_layer_call_fn_9336365?
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
M__inference_max_pooling3d_10_layer_call_and_return_conditional_losses_9336370
M__inference_max_pooling3d_10_layer_call_and_return_conditional_losses_9336375?
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
8__inference_batch_normalization_10_layer_call_fn_9336388
8__inference_batch_normalization_10_layer_call_fn_9336401
8__inference_batch_normalization_10_layer_call_fn_9336414
8__inference_batch_normalization_10_layer_call_fn_9336427?
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
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_9336445
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_9336463
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_9336481
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_9336499?
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
+__inference_conv3d_11_layer_call_fn_9336508?
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
F__inference_conv3d_11_layer_call_and_return_conditional_losses_9336519?
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
2__inference_max_pooling3d_11_layer_call_fn_9336524
2__inference_max_pooling3d_11_layer_call_fn_9336529?
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
M__inference_max_pooling3d_11_layer_call_and_return_conditional_losses_9336534
M__inference_max_pooling3d_11_layer_call_and_return_conditional_losses_9336539?
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
8__inference_batch_normalization_11_layer_call_fn_9336552
8__inference_batch_normalization_11_layer_call_fn_9336565
8__inference_batch_normalization_11_layer_call_fn_9336578
8__inference_batch_normalization_11_layer_call_fn_9336591?
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
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_9336609
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_9336627
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_9336645
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_9336663?
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
+__inference_conv3d_12_layer_call_fn_9336672?
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
F__inference_conv3d_12_layer_call_and_return_conditional_losses_9336683?
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
2__inference_max_pooling3d_12_layer_call_fn_9336688
2__inference_max_pooling3d_12_layer_call_fn_9336693?
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
M__inference_max_pooling3d_12_layer_call_and_return_conditional_losses_9336698
M__inference_max_pooling3d_12_layer_call_and_return_conditional_losses_9336703?
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
8__inference_batch_normalization_12_layer_call_fn_9336716
8__inference_batch_normalization_12_layer_call_fn_9336729
8__inference_batch_normalization_12_layer_call_fn_9336742
8__inference_batch_normalization_12_layer_call_fn_9336755?
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
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_9336773
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_9336791
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_9336809
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_9336827?
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
+__inference_conv3d_13_layer_call_fn_9336836?
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
F__inference_conv3d_13_layer_call_and_return_conditional_losses_9336847?
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
2__inference_max_pooling3d_13_layer_call_fn_9336852
2__inference_max_pooling3d_13_layer_call_fn_9336857?
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
M__inference_max_pooling3d_13_layer_call_and_return_conditional_losses_9336862
M__inference_max_pooling3d_13_layer_call_and_return_conditional_losses_9336867?
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
8__inference_batch_normalization_13_layer_call_fn_9336880
8__inference_batch_normalization_13_layer_call_fn_9336893
8__inference_batch_normalization_13_layer_call_fn_9336906
8__inference_batch_normalization_13_layer_call_fn_9336919?
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
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_9336937
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_9336955
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_9336973
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_9336991?
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
+__inference_conv3d_14_layer_call_fn_9337000?
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
F__inference_conv3d_14_layer_call_and_return_conditional_losses_9337011?
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
+__inference_1_dense18_layer_call_fn_9337020?
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
F__inference_1_dense18_layer_call_and_return_conditional_losses_9337031?
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
2__inference_max_pooling3d_14_layer_call_fn_9337036
2__inference_max_pooling3d_14_layer_call_fn_9337041?
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
M__inference_max_pooling3d_14_layer_call_and_return_conditional_losses_9337046
M__inference_max_pooling3d_14_layer_call_and_return_conditional_losses_9337051?
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
+__inference_2_dense32_layer_call_fn_9337060?
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
F__inference_2_dense32_layer_call_and_return_conditional_losses_9337071?
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
8__inference_batch_normalization_14_layer_call_fn_9337084
8__inference_batch_normalization_14_layer_call_fn_9337097
8__inference_batch_normalization_14_layer_call_fn_9337110
8__inference_batch_normalization_14_layer_call_fn_9337123?
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
S__inference_batch_normalization_14_layer_call_and_return_conditional_losses_9337141
S__inference_batch_normalization_14_layer_call_and_return_conditional_losses_9337159
S__inference_batch_normalization_14_layer_call_and_return_conditional_losses_9337177
S__inference_batch_normalization_14_layer_call_and_return_conditional_losses_9337195?
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
+__inference_3_dense64_layer_call_fn_9337204?
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
F__inference_3_dense64_layer_call_and_return_conditional_losses_9337215?
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
<__inference_global_average_pooling3d_2_layer_call_fn_9337220
<__inference_global_average_pooling3d_2_layer_call_fn_9337225?
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
W__inference_global_average_pooling3d_2_layer_call_and_return_conditional_losses_9337231
W__inference_global_average_pooling3d_2_layer_call_and_return_conditional_losses_9337237?
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
+__inference_4_dense32_layer_call_fn_9337246?
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
F__inference_4_dense32_layer_call_and_return_conditional_losses_9337257?
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
)__inference_dense_2_layer_call_fn_9337266?
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
D__inference_dense_2_layer_call_and_return_conditional_losses_9337277?
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
+__inference_5_dense18_layer_call_fn_9337286?
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
F__inference_5_dense18_layer_call_and_return_conditional_losses_9337297?
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
+__inference_dropout_2_layer_call_fn_9337302
+__inference_dropout_2_layer_call_fn_9337307?
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
F__inference_dropout_2_layer_call_and_return_conditional_losses_9337312
F__inference_dropout_2_layer_call_and_return_conditional_losses_9337324?
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
/__inference_concatenate_1_layer_call_fn_9337330?
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
J__inference_concatenate_1_layer_call_and_return_conditional_losses_9337337?
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
0__inference_combined_input_layer_call_fn_9337346?
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
K__inference_combined_input_layer_call_and_return_conditional_losses_9337357?
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
1__inference_ensemble_output_layer_call_fn_9337366?
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
L__inference_ensemble_output_layer_call_and_return_conditional_losses_9337377?
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
%__inference_signature_wrapper_9335780img3d_inputs
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
F__inference_1_dense18_layer_call_and_return_conditional_losses_9337031\uv/?,
%?"
 ?
inputs?????????

? "%?"
?
0?????????
? ~
+__inference_1_dense18_layer_call_fn_9337020Ouv/?,
%?"
 ?
inputs?????????

? "???????????
F__inference_2_dense32_layer_call_and_return_conditional_losses_9337071]?/?,
%?"
 ?
inputs?????????
? "%?"
?
0????????? 
? 
+__inference_2_dense32_layer_call_fn_9337060P?/?,
%?"
 ?
inputs?????????
? "?????????? ?
F__inference_3_dense64_layer_call_and_return_conditional_losses_9337215^??/?,
%?"
 ?
inputs????????? 
? "%?"
?
0?????????@
? ?
+__inference_3_dense64_layer_call_fn_9337204Q??/?,
%?"
 ?
inputs????????? 
? "??????????@?
F__inference_4_dense32_layer_call_and_return_conditional_losses_9337257^??/?,
%?"
 ?
inputs?????????@
? "%?"
?
0????????? 
? ?
+__inference_4_dense32_layer_call_fn_9337246Q??/?,
%?"
 ?
inputs?????????@
? "?????????? ?
F__inference_5_dense18_layer_call_and_return_conditional_losses_9337297^??/?,
%?"
 ?
inputs????????? 
? "%?"
?
0?????????
? ?
+__inference_5_dense18_layer_call_fn_9337286Q??/?,
%?"
 ?
inputs????????? 
? "???????????
"__inference__wrapped_model_9333325??#$./0167ABCDIJTUVW\]ghijopuv?????????????????n?k
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
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_9336445?./01Z?W
P?M
G?D
inputs8???????????????????????????????????? 
p 
? "L?I
B??
08???????????????????????????????????? 
? ?
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_9336463?./01Z?W
P?M
G?D
inputs8???????????????????????????????????? 
p
? "L?I
B??
08???????????????????????????????????? 
? ?
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_9336481z./01??<
5?2
,?)
inputs?????????@@ 
p 
? "1?.
'?$
0?????????@@ 
? ?
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_9336499z./01??<
5?2
,?)
inputs?????????@@ 
p
? "1?.
'?$
0?????????@@ 
? ?
8__inference_batch_normalization_10_layer_call_fn_9336388?./01Z?W
P?M
G?D
inputs8???????????????????????????????????? 
p 
? "??<8???????????????????????????????????? ?
8__inference_batch_normalization_10_layer_call_fn_9336401?./01Z?W
P?M
G?D
inputs8???????????????????????????????????? 
p
? "??<8???????????????????????????????????? ?
8__inference_batch_normalization_10_layer_call_fn_9336414m./01??<
5?2
,?)
inputs?????????@@ 
p 
? "$?!?????????@@ ?
8__inference_batch_normalization_10_layer_call_fn_9336427m./01??<
5?2
,?)
inputs?????????@@ 
p
? "$?!?????????@@ ?
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_9336609?ABCDZ?W
P?M
G?D
inputs8???????????????????????????????????? 
p 
? "L?I
B??
08???????????????????????????????????? 
? ?
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_9336627?ABCDZ?W
P?M
G?D
inputs8???????????????????????????????????? 
p
? "L?I
B??
08???????????????????????????????????? 
? ?
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_9336645zABCD??<
5?2
,?)
inputs?????????@@ 
p 
? "1?.
'?$
0?????????@@ 
? ?
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_9336663zABCD??<
5?2
,?)
inputs?????????@@ 
p
? "1?.
'?$
0?????????@@ 
? ?
8__inference_batch_normalization_11_layer_call_fn_9336552?ABCDZ?W
P?M
G?D
inputs8???????????????????????????????????? 
p 
? "??<8???????????????????????????????????? ?
8__inference_batch_normalization_11_layer_call_fn_9336565?ABCDZ?W
P?M
G?D
inputs8???????????????????????????????????? 
p
? "??<8???????????????????????????????????? ?
8__inference_batch_normalization_11_layer_call_fn_9336578mABCD??<
5?2
,?)
inputs?????????@@ 
p 
? "$?!?????????@@ ?
8__inference_batch_normalization_11_layer_call_fn_9336591mABCD??<
5?2
,?)
inputs?????????@@ 
p
? "$?!?????????@@ ?
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_9336773?TUVWZ?W
P?M
G?D
inputs8????????????????????????????????????@
p 
? "L?I
B??
08????????????????????????????????????@
? ?
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_9336791?TUVWZ?W
P?M
G?D
inputs8????????????????????????????????????@
p
? "L?I
B??
08????????????????????????????????????@
? ?
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_9336809zTUVW??<
5?2
,?)
inputs?????????@@@
p 
? "1?.
'?$
0?????????@@@
? ?
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_9336827zTUVW??<
5?2
,?)
inputs?????????@@@
p
? "1?.
'?$
0?????????@@@
? ?
8__inference_batch_normalization_12_layer_call_fn_9336716?TUVWZ?W
P?M
G?D
inputs8????????????????????????????????????@
p 
? "??<8????????????????????????????????????@?
8__inference_batch_normalization_12_layer_call_fn_9336729?TUVWZ?W
P?M
G?D
inputs8????????????????????????????????????@
p
? "??<8????????????????????????????????????@?
8__inference_batch_normalization_12_layer_call_fn_9336742mTUVW??<
5?2
,?)
inputs?????????@@@
p 
? "$?!?????????@@@?
8__inference_batch_normalization_12_layer_call_fn_9336755mTUVW??<
5?2
,?)
inputs?????????@@@
p
? "$?!?????????@@@?
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_9336937?ghij[?X
Q?N
H?E
inputs9?????????????????????????????????????
p 
? "M?J
C?@
09?????????????????????????????????????
? ?
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_9336955?ghij[?X
Q?N
H?E
inputs9?????????????????????????????????????
p
? "M?J
C?@
09?????????????????????????????????????
? ?
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_9336973|ghij@?=
6?3
-?*
inputs?????????@@?
p 
? "2?/
(?%
0?????????@@?
? ?
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_9336991|ghij@?=
6?3
-?*
inputs?????????@@?
p
? "2?/
(?%
0?????????@@?
? ?
8__inference_batch_normalization_13_layer_call_fn_9336880?ghij[?X
Q?N
H?E
inputs9?????????????????????????????????????
p 
? "@?=9??????????????????????????????????????
8__inference_batch_normalization_13_layer_call_fn_9336893?ghij[?X
Q?N
H?E
inputs9?????????????????????????????????????
p
? "@?=9??????????????????????????????????????
8__inference_batch_normalization_13_layer_call_fn_9336906oghij@?=
6?3
-?*
inputs?????????@@?
p 
? "%?"?????????@@??
8__inference_batch_normalization_13_layer_call_fn_9336919oghij@?=
6?3
-?*
inputs?????????@@?
p
? "%?"?????????@@??
S__inference_batch_normalization_14_layer_call_and_return_conditional_losses_9337141?????[?X
Q?N
H?E
inputs9?????????????????????????????????????
p 
? "M?J
C?@
09?????????????????????????????????????
? ?
S__inference_batch_normalization_14_layer_call_and_return_conditional_losses_9337159?????[?X
Q?N
H?E
inputs9?????????????????????????????????????
p
? "M?J
C?@
09?????????????????????????????????????
? ?
S__inference_batch_normalization_14_layer_call_and_return_conditional_losses_9337177?????@?=
6?3
-?*
inputs?????????@@?
p 
? "2?/
(?%
0?????????@@?
? ?
S__inference_batch_normalization_14_layer_call_and_return_conditional_losses_9337195?????@?=
6?3
-?*
inputs?????????@@?
p
? "2?/
(?%
0?????????@@?
? ?
8__inference_batch_normalization_14_layer_call_fn_9337084?????[?X
Q?N
H?E
inputs9?????????????????????????????????????
p 
? "@?=9??????????????????????????????????????
8__inference_batch_normalization_14_layer_call_fn_9337097?????[?X
Q?N
H?E
inputs9?????????????????????????????????????
p
? "@?=9??????????????????????????????????????
8__inference_batch_normalization_14_layer_call_fn_9337110s????@?=
6?3
-?*
inputs?????????@@?
p 
? "%?"?????????@@??
8__inference_batch_normalization_14_layer_call_fn_9337123s????@?=
6?3
-?*
inputs?????????@@?
p
? "%?"?????????@@??
K__inference_combined_input_layer_call_and_return_conditional_losses_9337357_??0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????	
? ?
0__inference_combined_input_layer_call_fn_9337346R??0?-
&?#
!?
inputs??????????
? "??????????	?
J__inference_concatenate_1_layer_call_and_return_conditional_losses_9337337?[?X
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
/__inference_concatenate_1_layer_call_fn_9337330x[?X
Q?N
L?I
"?
inputs/0?????????
#? 
inputs/1??????????
? "????????????
F__inference_conv3d_10_layer_call_and_return_conditional_losses_9336355x#$=?:
3?0
.?+
inputs???????????
? "3?0
)?&
0??????????? 
? ?
+__inference_conv3d_10_layer_call_fn_9336344k#$=?:
3?0
.?+
inputs???????????
? "&?#??????????? ?
F__inference_conv3d_11_layer_call_and_return_conditional_losses_9336519t67;?8
1?.
,?)
inputs?????????@@ 
? "1?.
'?$
0?????????@@ 
? ?
+__inference_conv3d_11_layer_call_fn_9336508g67;?8
1?.
,?)
inputs?????????@@ 
? "$?!?????????@@ ?
F__inference_conv3d_12_layer_call_and_return_conditional_losses_9336683tIJ;?8
1?.
,?)
inputs?????????@@ 
? "1?.
'?$
0?????????@@@
? ?
+__inference_conv3d_12_layer_call_fn_9336672gIJ;?8
1?.
,?)
inputs?????????@@ 
? "$?!?????????@@@?
F__inference_conv3d_13_layer_call_and_return_conditional_losses_9336847u\];?8
1?.
,?)
inputs?????????@@@
? "2?/
(?%
0?????????@@?
? ?
+__inference_conv3d_13_layer_call_fn_9336836h\];?8
1?.
,?)
inputs?????????@@@
? "%?"?????????@@??
F__inference_conv3d_14_layer_call_and_return_conditional_losses_9337011vop<?9
2?/
-?*
inputs?????????@@?
? "2?/
(?%
0?????????@@?
? ?
+__inference_conv3d_14_layer_call_fn_9337000iop<?9
2?/
-?*
inputs?????????@@?
? "%?"?????????@@??
D__inference_dense_2_layer_call_and_return_conditional_losses_9337277`??0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
)__inference_dense_2_layer_call_fn_9337266S??0?-
&?#
!?
inputs??????????
? "????????????
F__inference_dropout_2_layer_call_and_return_conditional_losses_9337312^4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ?
F__inference_dropout_2_layer_call_and_return_conditional_losses_9337324^4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? ?
+__inference_dropout_2_layer_call_fn_9337302Q4?1
*?'
!?
inputs??????????
p 
? "????????????
+__inference_dropout_2_layer_call_fn_9337307Q4?1
*?'
!?
inputs??????????
p
? "????????????
G__inference_ensemble4d_layer_call_and_return_conditional_losses_9335553??#$./0167ABCDIJTUVW\]ghijopuv?????????????????v?s
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
G__inference_ensemble4d_layer_call_and_return_conditional_losses_9335676??#$./0167ABCDIJTUVW\]ghijopuv?????????????????v?s
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
G__inference_ensemble4d_layer_call_and_return_conditional_losses_9336152??#$./0167ABCDIJTUVW\]ghijopuv?????????????????p?m
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
G__inference_ensemble4d_layer_call_and_return_conditional_losses_9336335??#$./0167ABCDIJTUVW\]ghijopuv?????????????????p?m
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
,__inference_ensemble4d_layer_call_fn_9334603??#$./0167ABCDIJTUVW\]ghijopuv?????????????????v?s
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
,__inference_ensemble4d_layer_call_fn_9335430??#$./0167ABCDIJTUVW\]ghijopuv?????????????????v?s
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
,__inference_ensemble4d_layer_call_fn_9335878??#$./0167ABCDIJTUVW\]ghijopuv?????????????????p?m
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
,__inference_ensemble4d_layer_call_fn_9335976??#$./0167ABCDIJTUVW\]ghijopuv?????????????????p?m
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
L__inference_ensemble_output_layer_call_and_return_conditional_losses_9337377^??/?,
%?"
 ?
inputs?????????	
? "%?"
?
0?????????
? ?
1__inference_ensemble_output_layer_call_fn_9337366Q??/?,
%?"
 ?
inputs?????????	
? "???????????
W__inference_global_average_pooling3d_2_layer_call_and_return_conditional_losses_9337231?_?\
U?R
P?M
inputsA?????????????????????????????????????????????
? ".?+
$?!
0??????????????????
? ?
W__inference_global_average_pooling3d_2_layer_call_and_return_conditional_losses_9337237f<?9
2?/
-?*
inputs?????????@@?
? "&?#
?
0??????????
? ?
<__inference_global_average_pooling3d_2_layer_call_fn_9337220?_?\
U?R
P?M
inputsA?????????????????????????????????????????????
? "!????????????????????
<__inference_global_average_pooling3d_2_layer_call_fn_9337225Y<?9
2?/
-?*
inputs?????????@@?
? "????????????
M__inference_max_pooling3d_10_layer_call_and_return_conditional_losses_9336370?_?\
U?R
P?M
inputsA?????????????????????????????????????????????
? "U?R
K?H
0A?????????????????????????????????????????????
? ?
M__inference_max_pooling3d_10_layer_call_and_return_conditional_losses_9336375r=?:
3?0
.?+
inputs??????????? 
? "1?.
'?$
0?????????@@ 
? ?
2__inference_max_pooling3d_10_layer_call_fn_9336360?_?\
U?R
P?M
inputsA?????????????????????????????????????????????
? "H?EA??????????????????????????????????????????????
2__inference_max_pooling3d_10_layer_call_fn_9336365e=?:
3?0
.?+
inputs??????????? 
? "$?!?????????@@ ?
M__inference_max_pooling3d_11_layer_call_and_return_conditional_losses_9336534?_?\
U?R
P?M
inputsA?????????????????????????????????????????????
? "U?R
K?H
0A?????????????????????????????????????????????
? ?
M__inference_max_pooling3d_11_layer_call_and_return_conditional_losses_9336539p;?8
1?.
,?)
inputs?????????@@ 
? "1?.
'?$
0?????????@@ 
? ?
2__inference_max_pooling3d_11_layer_call_fn_9336524?_?\
U?R
P?M
inputsA?????????????????????????????????????????????
? "H?EA??????????????????????????????????????????????
2__inference_max_pooling3d_11_layer_call_fn_9336529c;?8
1?.
,?)
inputs?????????@@ 
? "$?!?????????@@ ?
M__inference_max_pooling3d_12_layer_call_and_return_conditional_losses_9336698?_?\
U?R
P?M
inputsA?????????????????????????????????????????????
? "U?R
K?H
0A?????????????????????????????????????????????
? ?
M__inference_max_pooling3d_12_layer_call_and_return_conditional_losses_9336703p;?8
1?.
,?)
inputs?????????@@@
? "1?.
'?$
0?????????@@@
? ?
2__inference_max_pooling3d_12_layer_call_fn_9336688?_?\
U?R
P?M
inputsA?????????????????????????????????????????????
? "H?EA??????????????????????????????????????????????
2__inference_max_pooling3d_12_layer_call_fn_9336693c;?8
1?.
,?)
inputs?????????@@@
? "$?!?????????@@@?
M__inference_max_pooling3d_13_layer_call_and_return_conditional_losses_9336862?_?\
U?R
P?M
inputsA?????????????????????????????????????????????
? "U?R
K?H
0A?????????????????????????????????????????????
? ?
M__inference_max_pooling3d_13_layer_call_and_return_conditional_losses_9336867r<?9
2?/
-?*
inputs?????????@@?
? "2?/
(?%
0?????????@@?
? ?
2__inference_max_pooling3d_13_layer_call_fn_9336852?_?\
U?R
P?M
inputsA?????????????????????????????????????????????
? "H?EA??????????????????????????????????????????????
2__inference_max_pooling3d_13_layer_call_fn_9336857e<?9
2?/
-?*
inputs?????????@@?
? "%?"?????????@@??
M__inference_max_pooling3d_14_layer_call_and_return_conditional_losses_9337046?_?\
U?R
P?M
inputsA?????????????????????????????????????????????
? "U?R
K?H
0A?????????????????????????????????????????????
? ?
M__inference_max_pooling3d_14_layer_call_and_return_conditional_losses_9337051r<?9
2?/
-?*
inputs?????????@@?
? "2?/
(?%
0?????????@@?
? ?
2__inference_max_pooling3d_14_layer_call_fn_9337036?_?\
U?R
P?M
inputsA?????????????????????????????????????????????
? "H?EA??????????????????????????????????????????????
2__inference_max_pooling3d_14_layer_call_fn_9337041e<?9
2?/
-?*
inputs?????????@@?
? "%?"?????????@@??
%__inference_signature_wrapper_9335780??#$./0167ABCDIJTUVW\]ghijopuv????????????????????
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
ры
—£
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
Њ
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
executor_typestring И
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.3.02v2.3.0-rc2-23-gb36436b0878ђС	
|
conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1/kernel
u
 conv1/kernel/Read/ReadVariableOpReadVariableOpconv1/kernel*&
_output_shapes
:*
dtype0
l

conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
conv1/bias
e
conv1/bias/Read/ReadVariableOpReadVariableOp
conv1/bias*
_output_shapes
:*
dtype0
|
conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_nameconv2/kernel
u
 conv2/kernel/Read/ReadVariableOpReadVariableOpconv2/kernel*&
_output_shapes
:
*
dtype0
l

conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_name
conv2/bias
e
conv2/bias/Read/ReadVariableOpReadVariableOp
conv2/bias*
_output_shapes
:
*
dtype0
|
conv3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_nameconv3/kernel
u
 conv3/kernel/Read/ReadVariableOpReadVariableOpconv3/kernel*&
_output_shapes
:
*
dtype0
l

conv3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
conv3/bias
e
conv3/bias/Read/ReadVariableOpReadVariableOp
conv3/bias*
_output_shapes
:*
dtype0
А
deconv3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedeconv3/kernel
y
"deconv3/kernel/Read/ReadVariableOpReadVariableOpdeconv3/kernel*&
_output_shapes
:
*
dtype0
p
deconv3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedeconv3/bias
i
 deconv3/bias/Read/ReadVariableOpReadVariableOpdeconv3/bias*
_output_shapes
:
*
dtype0
А
deconv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedeconv2/kernel
y
"deconv2/kernel/Read/ReadVariableOpReadVariableOpdeconv2/kernel*&
_output_shapes
:
*
dtype0
p
deconv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedeconv2/bias
i
 deconv2/bias/Read/ReadVariableOpReadVariableOpdeconv2/bias*
_output_shapes
:*
dtype0
А
deconv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedeconv1/kernel
y
"deconv1/kernel/Read/ReadVariableOpReadVariableOpdeconv1/kernel*&
_output_shapes
:*
dtype0
p
deconv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedeconv1/bias
i
 deconv1/bias/Read/ReadVariableOpReadVariableOpdeconv1/bias*
_output_shapes
:*
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
К
Adam/conv3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*$
shared_nameAdam/conv3/kernel/m
Г
'Adam/conv3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3/kernel/m*&
_output_shapes
:
*
dtype0
z
Adam/conv3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/conv3/bias/m
s
%Adam/conv3/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3/bias/m*
_output_shapes
:*
dtype0
О
Adam/deconv3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/deconv3/kernel/m
З
)Adam/deconv3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/deconv3/kernel/m*&
_output_shapes
:
*
dtype0
~
Adam/deconv3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*$
shared_nameAdam/deconv3/bias/m
w
'Adam/deconv3/bias/m/Read/ReadVariableOpReadVariableOpAdam/deconv3/bias/m*
_output_shapes
:
*
dtype0
К
Adam/conv3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*$
shared_nameAdam/conv3/kernel/v
Г
'Adam/conv3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3/kernel/v*&
_output_shapes
:
*
dtype0
z
Adam/conv3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/conv3/bias/v
s
%Adam/conv3/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3/bias/v*
_output_shapes
:*
dtype0
О
Adam/deconv3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/deconv3/kernel/v
З
)Adam/deconv3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/deconv3/kernel/v*&
_output_shapes
:
*
dtype0
~
Adam/deconv3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*$
shared_nameAdam/deconv3/bias/v
w
'Adam/deconv3/bias/v/Read/ReadVariableOpReadVariableOpAdam/deconv3/bias/v*
_output_shapes
:
*
dtype0

NoOpNoOp
”;
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*О;
valueД;BБ; Bъ:
Ђ
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer-9
layer_with_weights-5
layer-10
layer-11
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
Н

kernel
bias
#_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api
R
regularization_losses
	variables
trainable_variables
	keras_api
Н

kernel
bias
# _self_saveable_object_factories
!regularization_losses
"	variables
#trainable_variables
$	keras_api
R
%regularization_losses
&	variables
'trainable_variables
(	keras_api
h

)kernel
*bias
+regularization_losses
,	variables
-trainable_variables
.	keras_api
R
/regularization_losses
0	variables
1trainable_variables
2	keras_api
h

3kernel
4bias
5regularization_losses
6	variables
7trainable_variables
8	keras_api
R
9regularization_losses
:	variables
;trainable_variables
<	keras_api
Н

=kernel
>bias
#?_self_saveable_object_factories
@regularization_losses
A	variables
Btrainable_variables
C	keras_api
R
Dregularization_losses
E	variables
Ftrainable_variables
G	keras_api
Н

Hkernel
Ibias
#J_self_saveable_object_factories
Kregularization_losses
L	variables
Mtrainable_variables
N	keras_api
R
Oregularization_losses
P	variables
Qtrainable_variables
R	keras_api
Р
Siter

Tbeta_1

Ubeta_2
	Vdecay
Wlearning_rate)mЮ*mЯ3m†4m°)vҐ*v£3v§4v•
 
V
0
1
2
3
)4
*5
36
47
=8
>9
H10
I11

)0
*1
32
43
≠
Xlayer_metrics
Ymetrics
regularization_losses
	variables
Zlayer_regularization_losses

[layers
\non_trainable_variables
trainable_variables
 
XV
VARIABLE_VALUEconv1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
conv1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

0
1
 
≠
]layer_metrics
^metrics
regularization_losses
_layer_regularization_losses
	variables

`layers
anon_trainable_variables
trainable_variables
 
 
 
≠
blayer_metrics
cmetrics
regularization_losses
dlayer_regularization_losses
	variables

elayers
fnon_trainable_variables
trainable_variables
XV
VARIABLE_VALUEconv2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
conv2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

0
1
 
≠
glayer_metrics
hmetrics
!regularization_losses
ilayer_regularization_losses
"	variables

jlayers
knon_trainable_variables
#trainable_variables
 
 
 
≠
llayer_metrics
mmetrics
%regularization_losses
nlayer_regularization_losses
&	variables

olayers
pnon_trainable_variables
'trainable_variables
XV
VARIABLE_VALUEconv3/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
conv3/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

)0
*1

)0
*1
≠
qlayer_metrics
rmetrics
+regularization_losses
slayer_regularization_losses
,	variables

tlayers
unon_trainable_variables
-trainable_variables
 
 
 
≠
vlayer_metrics
wmetrics
/regularization_losses
xlayer_regularization_losses
0	variables

ylayers
znon_trainable_variables
1trainable_variables
ZX
VARIABLE_VALUEdeconv3/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdeconv3/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

30
41

30
41
≠
{layer_metrics
|metrics
5regularization_losses
}layer_regularization_losses
6	variables

~layers
non_trainable_variables
7trainable_variables
 
 
 
≤
Аlayer_metrics
Бmetrics
9regularization_losses
 Вlayer_regularization_losses
:	variables
Гlayers
Дnon_trainable_variables
;trainable_variables
ZX
VARIABLE_VALUEdeconv2/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdeconv2/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

=0
>1
 
≤
Еlayer_metrics
Жmetrics
@regularization_losses
 Зlayer_regularization_losses
A	variables
Иlayers
Йnon_trainable_variables
Btrainable_variables
 
 
 
≤
Кlayer_metrics
Лmetrics
Dregularization_losses
 Мlayer_regularization_losses
E	variables
Нlayers
Оnon_trainable_variables
Ftrainable_variables
ZX
VARIABLE_VALUEdeconv1/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdeconv1/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

H0
I1
 
≤
Пlayer_metrics
Рmetrics
Kregularization_losses
 Сlayer_regularization_losses
L	variables
Тlayers
Уnon_trainable_variables
Mtrainable_variables
 
 
 
≤
Фlayer_metrics
Хmetrics
Oregularization_losses
 Цlayer_regularization_losses
P	variables
Чlayers
Шnon_trainable_variables
Qtrainable_variables
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

Щ0
 
V
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
8
0
1
2
3
=4
>5
H6
I7
 
 
 
 

0
1
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
0
1
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

=0
>1
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
H0
I1
 
 
 
 
 
8

Ъtotal

Ыcount
Ь	variables
Э	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

Ъ0
Ы1

Ь	variables
{y
VARIABLE_VALUEAdam/conv3/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv3/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/deconv3/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/deconv3/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv3/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv3/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/deconv3/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/deconv3/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
К
serving_default_input_2Placeholder*/
_output_shapes
:€€€€€€€€€@@*
dtype0*$
shape:€€€€€€€€€@@
ь
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_2conv1/kernel
conv1/biasconv2/kernel
conv2/biasconv3/kernel
conv3/biasdeconv3/kerneldeconv3/biasdeconv2/kerneldeconv2/biasdeconv1/kerneldeconv1/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@@*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *,
f'R%
#__inference_signature_wrapper_41272
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ч	
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename conv1/kernel/Read/ReadVariableOpconv1/bias/Read/ReadVariableOp conv2/kernel/Read/ReadVariableOpconv2/bias/Read/ReadVariableOp conv3/kernel/Read/ReadVariableOpconv3/bias/Read/ReadVariableOp"deconv3/kernel/Read/ReadVariableOp deconv3/bias/Read/ReadVariableOp"deconv2/kernel/Read/ReadVariableOp deconv2/bias/Read/ReadVariableOp"deconv1/kernel/Read/ReadVariableOp deconv1/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp'Adam/conv3/kernel/m/Read/ReadVariableOp%Adam/conv3/bias/m/Read/ReadVariableOp)Adam/deconv3/kernel/m/Read/ReadVariableOp'Adam/deconv3/bias/m/Read/ReadVariableOp'Adam/conv3/kernel/v/Read/ReadVariableOp%Adam/conv3/bias/v/Read/ReadVariableOp)Adam/deconv3/kernel/v/Read/ReadVariableOp'Adam/deconv3/bias/v/Read/ReadVariableOpConst*(
Tin!
2	*
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
GPU2*0J 8В *'
f"R 
__inference__traced_save_41721
÷
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv1/kernel
conv1/biasconv2/kernel
conv2/biasconv3/kernel
conv3/biasdeconv3/kerneldeconv3/biasdeconv2/kerneldeconv2/biasdeconv1/kerneldeconv1/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/conv3/kernel/mAdam/conv3/bias/mAdam/deconv3/kernel/mAdam/deconv3/bias/mAdam/conv3/kernel/vAdam/conv3/bias/vAdam/deconv3/kernel/vAdam/deconv3/bias/v*'
Tin 
2*
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
GPU2*0J 8В **
f%R#
!__inference__traced_restore_41812„О
п
b
F__inference_conv2-leaky_layer_call_and_return_conditional_losses_40952

inputs
identityl
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:€€€€€€€€€
*
alpha%Ќћћ=2
	LeakyRelus
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€
2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€
:W S
/
_output_shapes
:€€€€€€€€€

 
_user_specified_nameinputs
Ґ
®
@__inference_conv1_layer_call_and_return_conditional_losses_40892

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€  *
paddingVALID*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€  2	
BiasAddl
IdentityIdentityBiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€  2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€@@:::W S
/
_output_shapes
:€€€€€€€€€@@
 
_user_specified_nameinputs
≠3
З
I__inference_autoencoder-v3_layer_call_and_return_conditional_losses_41094
input_2
conv1_41057
conv1_41059
conv2_41063
conv2_41065
conv3_41069
conv3_41071
deconv3_41075
deconv3_41077
deconv2_41081
deconv2_41083
deconv1_41087
deconv1_41089
identityИҐconv1/StatefulPartitionedCallҐconv2/StatefulPartitionedCallҐconv3/StatefulPartitionedCallҐdeconv1/StatefulPartitionedCallҐdeconv2/StatefulPartitionedCallҐdeconv3/StatefulPartitionedCallО
conv1/StatefulPartitionedCallStatefulPartitionedCallinput_2conv1_41057conv1_41059*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_conv1_layer_call_and_return_conditional_losses_408922
conv1/StatefulPartitionedCallЗ
conv1-leaky/PartitionedCallPartitionedCall&conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€  * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_conv1-leaky_layer_call_and_return_conditional_losses_409132
conv1-leaky/PartitionedCallЂ
conv2/StatefulPartitionedCallStatefulPartitionedCall$conv1-leaky/PartitionedCall:output:0conv2_41063conv2_41065*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_conv2_layer_call_and_return_conditional_losses_409312
conv2/StatefulPartitionedCallЗ
conv2-leaky/PartitionedCallPartitionedCall&conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_conv2-leaky_layer_call_and_return_conditional_losses_409522
conv2-leaky/PartitionedCallЂ
conv3/StatefulPartitionedCallStatefulPartitionedCall$conv2-leaky/PartitionedCall:output:0conv3_41069conv3_41071*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_conv3_layer_call_and_return_conditional_losses_409702
conv3/StatefulPartitionedCallЗ
conv3-leaky/PartitionedCallPartitionedCall&conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_conv3-leaky_layer_call_and_return_conditional_losses_409912
conv3-leaky/PartitionedCall«
deconv3/StatefulPartitionedCallStatefulPartitionedCall$conv3-leaky/PartitionedCall:output:0deconv3_41075deconv3_41077*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_deconv3_layer_call_and_return_conditional_losses_407722!
deconv3/StatefulPartitionedCall°
deconv3-leaky/PartitionedCallPartitionedCall(deconv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_deconv3-leaky_layer_call_and_return_conditional_losses_410092
deconv3-leaky/PartitionedCall…
deconv2/StatefulPartitionedCallStatefulPartitionedCall&deconv3-leaky/PartitionedCall:output:0deconv2_41081deconv2_41083*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_deconv2_layer_call_and_return_conditional_losses_408202!
deconv2/StatefulPartitionedCall°
deconv2-leaky/PartitionedCallPartitionedCall(deconv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_deconv2-leaky_layer_call_and_return_conditional_losses_410272
deconv2-leaky/PartitionedCall…
deconv1/StatefulPartitionedCallStatefulPartitionedCall&deconv2-leaky/PartitionedCall:output:0deconv1_41087deconv1_41089*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_deconv1_layer_call_and_return_conditional_losses_408682!
deconv1/StatefulPartitionedCallІ
deconv1-sigmoid/PartitionedCallPartitionedCall(deconv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_deconv1-sigmoid_layer_call_and_return_conditional_losses_410452!
deconv1-sigmoid/PartitionedCall№
IdentityIdentity(deconv1-sigmoid/PartitionedCall:output:0^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^conv3/StatefulPartitionedCall ^deconv1/StatefulPartitionedCall ^deconv2/StatefulPartitionedCall ^deconv3/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:€€€€€€€€€@@::::::::::::2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2>
conv3/StatefulPartitionedCallconv3/StatefulPartitionedCall2B
deconv1/StatefulPartitionedCalldeconv1/StatefulPartitionedCall2B
deconv2/StatefulPartitionedCalldeconv2/StatefulPartitionedCall2B
deconv3/StatefulPartitionedCalldeconv3/StatefulPartitionedCall:X T
/
_output_shapes
:€€€€€€€€€@@
!
_user_specified_name	input_2
є
G
+__inference_conv1-leaky_layer_call_fn_41529

inputs
identityѕ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€  * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_conv1-leaky_layer_call_and_return_conditional_losses_409132
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:€€€€€€€€€  2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€  :W S
/
_output_shapes
:€€€€€€€€€  
 
_user_specified_nameinputs
у	
Ы
.__inference_autoencoder-v3_layer_call_fn_41233
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identityИҐStatefulPartitionedCallЪ
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_autoencoder-v3_layer_call_and_return_conditional_losses_412062
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:€€€€€€€€€@@::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:€€€€€€€€€@@
!
_user_specified_name	input_2
™3
Ж
I__inference_autoencoder-v3_layer_call_and_return_conditional_losses_41206

inputs
conv1_41169
conv1_41171
conv2_41175
conv2_41177
conv3_41181
conv3_41183
deconv3_41187
deconv3_41189
deconv2_41193
deconv2_41195
deconv1_41199
deconv1_41201
identityИҐconv1/StatefulPartitionedCallҐconv2/StatefulPartitionedCallҐconv3/StatefulPartitionedCallҐdeconv1/StatefulPartitionedCallҐdeconv2/StatefulPartitionedCallҐdeconv3/StatefulPartitionedCallН
conv1/StatefulPartitionedCallStatefulPartitionedCallinputsconv1_41169conv1_41171*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_conv1_layer_call_and_return_conditional_losses_408922
conv1/StatefulPartitionedCallЗ
conv1-leaky/PartitionedCallPartitionedCall&conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€  * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_conv1-leaky_layer_call_and_return_conditional_losses_409132
conv1-leaky/PartitionedCallЂ
conv2/StatefulPartitionedCallStatefulPartitionedCall$conv1-leaky/PartitionedCall:output:0conv2_41175conv2_41177*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_conv2_layer_call_and_return_conditional_losses_409312
conv2/StatefulPartitionedCallЗ
conv2-leaky/PartitionedCallPartitionedCall&conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_conv2-leaky_layer_call_and_return_conditional_losses_409522
conv2-leaky/PartitionedCallЂ
conv3/StatefulPartitionedCallStatefulPartitionedCall$conv2-leaky/PartitionedCall:output:0conv3_41181conv3_41183*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_conv3_layer_call_and_return_conditional_losses_409702
conv3/StatefulPartitionedCallЗ
conv3-leaky/PartitionedCallPartitionedCall&conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_conv3-leaky_layer_call_and_return_conditional_losses_409912
conv3-leaky/PartitionedCall«
deconv3/StatefulPartitionedCallStatefulPartitionedCall$conv3-leaky/PartitionedCall:output:0deconv3_41187deconv3_41189*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_deconv3_layer_call_and_return_conditional_losses_407722!
deconv3/StatefulPartitionedCall°
deconv3-leaky/PartitionedCallPartitionedCall(deconv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_deconv3-leaky_layer_call_and_return_conditional_losses_410092
deconv3-leaky/PartitionedCall…
deconv2/StatefulPartitionedCallStatefulPartitionedCall&deconv3-leaky/PartitionedCall:output:0deconv2_41193deconv2_41195*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_deconv2_layer_call_and_return_conditional_losses_408202!
deconv2/StatefulPartitionedCall°
deconv2-leaky/PartitionedCallPartitionedCall(deconv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_deconv2-leaky_layer_call_and_return_conditional_losses_410272
deconv2-leaky/PartitionedCall…
deconv1/StatefulPartitionedCallStatefulPartitionedCall&deconv2-leaky/PartitionedCall:output:0deconv1_41199deconv1_41201*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_deconv1_layer_call_and_return_conditional_losses_408682!
deconv1/StatefulPartitionedCallІ
deconv1-sigmoid/PartitionedCallPartitionedCall(deconv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_deconv1-sigmoid_layer_call_and_return_conditional_losses_410452!
deconv1-sigmoid/PartitionedCall№
IdentityIdentity(deconv1-sigmoid/PartitionedCall:output:0^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^conv3/StatefulPartitionedCall ^deconv1/StatefulPartitionedCall ^deconv2/StatefulPartitionedCall ^deconv3/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:€€€€€€€€€@@::::::::::::2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2>
conv3/StatefulPartitionedCallconv3/StatefulPartitionedCall2B
deconv1/StatefulPartitionedCalldeconv1/StatefulPartitionedCall2B
deconv2/StatefulPartitionedCalldeconv2/StatefulPartitionedCall2B
deconv3/StatefulPartitionedCalldeconv3/StatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€@@
 
_user_specified_nameinputs
Ґ
®
@__inference_conv2_layer_call_and_return_conditional_losses_41539

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€
*
paddingVALID*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€
2	
BiasAddl
IdentityIdentityBiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€  :::W S
/
_output_shapes
:€€€€€€€€€  
 
_user_specified_nameinputs
¬$
і
B__inference_deconv1_layer_call_and_return_conditional_losses_40868

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identityИD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2в
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2м
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2м
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulP
add/yConst*
_output_shapes
: *
dtype0*
value	B : 2
add/yM
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: 2
addT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
add_1/yConst*
_output_shapes
: *
dtype0*
value	B : 2	
add_1/yU
add_1AddV2	mul_1:z:0add_1/y:output:0*
T0*
_output_shapes
: 2
add_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3В
stackPackstrided_slice:output:0add:z:0	add_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2м
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3≥
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_transpose/ReadVariableOpс
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*
paddingVALID*
strides
2
conv2d_transposeМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp§
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2	
BiasAdd~
IdentityIdentityBiasAdd:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:::i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Ґ
®
@__inference_conv2_layer_call_and_return_conditional_losses_40931

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€
*
paddingVALID*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€
2	
BiasAddl
IdentityIdentityBiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€  :::W S
/
_output_shapes
:€€€€€€€€€  
 
_user_specified_nameinputs
ч
z
%__inference_conv3_layer_call_fn_41577

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_conv3_layer_call_and_return_conditional_losses_409702
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€
::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€

 
_user_specified_nameinputs
у	
Ы
.__inference_autoencoder-v3_layer_call_fn_41164
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identityИҐStatefulPartitionedCallЪ
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_autoencoder-v3_layer_call_and_return_conditional_losses_411372
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:€€€€€€€€€@@::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:€€€€€€€€€@@
!
_user_specified_name	input_2
К
K
/__inference_deconv1-sigmoid_layer_call_fn_41617

inputs
identityе
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_deconv1-sigmoid_layer_call_and_return_conditional_losses_410452
PartitionedCallЖ
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
“}
Ь
 __inference__wrapped_model_40734
input_27
3autoencoder_v3_conv1_conv2d_readvariableop_resource8
4autoencoder_v3_conv1_biasadd_readvariableop_resource7
3autoencoder_v3_conv2_conv2d_readvariableop_resource8
4autoencoder_v3_conv2_biasadd_readvariableop_resource7
3autoencoder_v3_conv3_conv2d_readvariableop_resource8
4autoencoder_v3_conv3_biasadd_readvariableop_resourceC
?autoencoder_v3_deconv3_conv2d_transpose_readvariableop_resource:
6autoencoder_v3_deconv3_biasadd_readvariableop_resourceC
?autoencoder_v3_deconv2_conv2d_transpose_readvariableop_resource:
6autoencoder_v3_deconv2_biasadd_readvariableop_resourceC
?autoencoder_v3_deconv1_conv2d_transpose_readvariableop_resource:
6autoencoder_v3_deconv1_biasadd_readvariableop_resource
identityИ‘
*autoencoder-v3/conv1/Conv2D/ReadVariableOpReadVariableOp3autoencoder_v3_conv1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02,
*autoencoder-v3/conv1/Conv2D/ReadVariableOpд
autoencoder-v3/conv1/Conv2DConv2Dinput_22autoencoder-v3/conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€  *
paddingVALID*
strides
2
autoencoder-v3/conv1/Conv2DЋ
+autoencoder-v3/conv1/BiasAdd/ReadVariableOpReadVariableOp4autoencoder_v3_conv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+autoencoder-v3/conv1/BiasAdd/ReadVariableOp№
autoencoder-v3/conv1/BiasAddBiasAdd$autoencoder-v3/conv1/Conv2D:output:03autoencoder-v3/conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€  2
autoencoder-v3/conv1/BiasAddЅ
$autoencoder-v3/conv1-leaky/LeakyRelu	LeakyRelu%autoencoder-v3/conv1/BiasAdd:output:0*/
_output_shapes
:€€€€€€€€€  *
alpha%Ќћћ=2&
$autoencoder-v3/conv1-leaky/LeakyRelu‘
*autoencoder-v3/conv2/Conv2D/ReadVariableOpReadVariableOp3autoencoder_v3_conv2_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02,
*autoencoder-v3/conv2/Conv2D/ReadVariableOpП
autoencoder-v3/conv2/Conv2DConv2D2autoencoder-v3/conv1-leaky/LeakyRelu:activations:02autoencoder-v3/conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€
*
paddingVALID*
strides
2
autoencoder-v3/conv2/Conv2DЋ
+autoencoder-v3/conv2/BiasAdd/ReadVariableOpReadVariableOp4autoencoder_v3_conv2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02-
+autoencoder-v3/conv2/BiasAdd/ReadVariableOp№
autoencoder-v3/conv2/BiasAddBiasAdd$autoencoder-v3/conv2/Conv2D:output:03autoencoder-v3/conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€
2
autoencoder-v3/conv2/BiasAddЅ
$autoencoder-v3/conv2-leaky/LeakyRelu	LeakyRelu%autoencoder-v3/conv2/BiasAdd:output:0*/
_output_shapes
:€€€€€€€€€
*
alpha%Ќћћ=2&
$autoencoder-v3/conv2-leaky/LeakyRelu‘
*autoencoder-v3/conv3/Conv2D/ReadVariableOpReadVariableOp3autoencoder_v3_conv3_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02,
*autoencoder-v3/conv3/Conv2D/ReadVariableOpП
autoencoder-v3/conv3/Conv2DConv2D2autoencoder-v3/conv2-leaky/LeakyRelu:activations:02autoencoder-v3/conv3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingVALID*
strides
2
autoencoder-v3/conv3/Conv2DЋ
+autoencoder-v3/conv3/BiasAdd/ReadVariableOpReadVariableOp4autoencoder_v3_conv3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+autoencoder-v3/conv3/BiasAdd/ReadVariableOp№
autoencoder-v3/conv3/BiasAddBiasAdd$autoencoder-v3/conv3/Conv2D:output:03autoencoder-v3/conv3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2
autoencoder-v3/conv3/BiasAddЅ
$autoencoder-v3/conv3-leaky/LeakyRelu	LeakyRelu%autoencoder-v3/conv3/BiasAdd:output:0*/
_output_shapes
:€€€€€€€€€*
alpha%Ќћћ=2&
$autoencoder-v3/conv3-leaky/LeakyReluЮ
autoencoder-v3/deconv3/ShapeShape2autoencoder-v3/conv3-leaky/LeakyRelu:activations:0*
T0*
_output_shapes
:2
autoencoder-v3/deconv3/ShapeҐ
*autoencoder-v3/deconv3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*autoencoder-v3/deconv3/strided_slice/stack¶
,autoencoder-v3/deconv3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,autoencoder-v3/deconv3/strided_slice/stack_1¶
,autoencoder-v3/deconv3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,autoencoder-v3/deconv3/strided_slice/stack_2м
$autoencoder-v3/deconv3/strided_sliceStridedSlice%autoencoder-v3/deconv3/Shape:output:03autoencoder-v3/deconv3/strided_slice/stack:output:05autoencoder-v3/deconv3/strided_slice/stack_1:output:05autoencoder-v3/deconv3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$autoencoder-v3/deconv3/strided_sliceВ
autoencoder-v3/deconv3/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2 
autoencoder-v3/deconv3/stack/1В
autoencoder-v3/deconv3/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2 
autoencoder-v3/deconv3/stack/2В
autoencoder-v3/deconv3/stack/3Const*
_output_shapes
: *
dtype0*
value	B :
2 
autoencoder-v3/deconv3/stack/3Ь
autoencoder-v3/deconv3/stackPack-autoencoder-v3/deconv3/strided_slice:output:0'autoencoder-v3/deconv3/stack/1:output:0'autoencoder-v3/deconv3/stack/2:output:0'autoencoder-v3/deconv3/stack/3:output:0*
N*
T0*
_output_shapes
:2
autoencoder-v3/deconv3/stack¶
,autoencoder-v3/deconv3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,autoencoder-v3/deconv3/strided_slice_1/stack™
.autoencoder-v3/deconv3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.autoencoder-v3/deconv3/strided_slice_1/stack_1™
.autoencoder-v3/deconv3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.autoencoder-v3/deconv3/strided_slice_1/stack_2ц
&autoencoder-v3/deconv3/strided_slice_1StridedSlice%autoencoder-v3/deconv3/stack:output:05autoencoder-v3/deconv3/strided_slice_1/stack:output:07autoencoder-v3/deconv3/strided_slice_1/stack_1:output:07autoencoder-v3/deconv3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&autoencoder-v3/deconv3/strided_slice_1ш
6autoencoder-v3/deconv3/conv2d_transpose/ReadVariableOpReadVariableOp?autoencoder_v3_deconv3_conv2d_transpose_readvariableop_resource*&
_output_shapes
:
*
dtype028
6autoencoder-v3/deconv3/conv2d_transpose/ReadVariableOpз
'autoencoder-v3/deconv3/conv2d_transposeConv2DBackpropInput%autoencoder-v3/deconv3/stack:output:0>autoencoder-v3/deconv3/conv2d_transpose/ReadVariableOp:value:02autoencoder-v3/conv3-leaky/LeakyRelu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€
*
paddingVALID*
strides
2)
'autoencoder-v3/deconv3/conv2d_transpose—
-autoencoder-v3/deconv3/BiasAdd/ReadVariableOpReadVariableOp6autoencoder_v3_deconv3_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02/
-autoencoder-v3/deconv3/BiasAdd/ReadVariableOpо
autoencoder-v3/deconv3/BiasAddBiasAdd0autoencoder-v3/deconv3/conv2d_transpose:output:05autoencoder-v3/deconv3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€
2 
autoencoder-v3/deconv3/BiasAdd«
&autoencoder-v3/deconv3-leaky/LeakyRelu	LeakyRelu'autoencoder-v3/deconv3/BiasAdd:output:0*/
_output_shapes
:€€€€€€€€€
*
alpha%Ќћћ=2(
&autoencoder-v3/deconv3-leaky/LeakyRelu†
autoencoder-v3/deconv2/ShapeShape4autoencoder-v3/deconv3-leaky/LeakyRelu:activations:0*
T0*
_output_shapes
:2
autoencoder-v3/deconv2/ShapeҐ
*autoencoder-v3/deconv2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*autoencoder-v3/deconv2/strided_slice/stack¶
,autoencoder-v3/deconv2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,autoencoder-v3/deconv2/strided_slice/stack_1¶
,autoencoder-v3/deconv2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,autoencoder-v3/deconv2/strided_slice/stack_2м
$autoencoder-v3/deconv2/strided_sliceStridedSlice%autoencoder-v3/deconv2/Shape:output:03autoencoder-v3/deconv2/strided_slice/stack:output:05autoencoder-v3/deconv2/strided_slice/stack_1:output:05autoencoder-v3/deconv2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$autoencoder-v3/deconv2/strided_sliceВ
autoencoder-v3/deconv2/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 2 
autoencoder-v3/deconv2/stack/1В
autoencoder-v3/deconv2/stack/2Const*
_output_shapes
: *
dtype0*
value	B : 2 
autoencoder-v3/deconv2/stack/2В
autoencoder-v3/deconv2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2 
autoencoder-v3/deconv2/stack/3Ь
autoencoder-v3/deconv2/stackPack-autoencoder-v3/deconv2/strided_slice:output:0'autoencoder-v3/deconv2/stack/1:output:0'autoencoder-v3/deconv2/stack/2:output:0'autoencoder-v3/deconv2/stack/3:output:0*
N*
T0*
_output_shapes
:2
autoencoder-v3/deconv2/stack¶
,autoencoder-v3/deconv2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,autoencoder-v3/deconv2/strided_slice_1/stack™
.autoencoder-v3/deconv2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.autoencoder-v3/deconv2/strided_slice_1/stack_1™
.autoencoder-v3/deconv2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.autoencoder-v3/deconv2/strided_slice_1/stack_2ц
&autoencoder-v3/deconv2/strided_slice_1StridedSlice%autoencoder-v3/deconv2/stack:output:05autoencoder-v3/deconv2/strided_slice_1/stack:output:07autoencoder-v3/deconv2/strided_slice_1/stack_1:output:07autoencoder-v3/deconv2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&autoencoder-v3/deconv2/strided_slice_1ш
6autoencoder-v3/deconv2/conv2d_transpose/ReadVariableOpReadVariableOp?autoencoder_v3_deconv2_conv2d_transpose_readvariableop_resource*&
_output_shapes
:
*
dtype028
6autoencoder-v3/deconv2/conv2d_transpose/ReadVariableOpй
'autoencoder-v3/deconv2/conv2d_transposeConv2DBackpropInput%autoencoder-v3/deconv2/stack:output:0>autoencoder-v3/deconv2/conv2d_transpose/ReadVariableOp:value:04autoencoder-v3/deconv3-leaky/LeakyRelu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€  *
paddingVALID*
strides
2)
'autoencoder-v3/deconv2/conv2d_transpose—
-autoencoder-v3/deconv2/BiasAdd/ReadVariableOpReadVariableOp6autoencoder_v3_deconv2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-autoencoder-v3/deconv2/BiasAdd/ReadVariableOpо
autoencoder-v3/deconv2/BiasAddBiasAdd0autoencoder-v3/deconv2/conv2d_transpose:output:05autoencoder-v3/deconv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€  2 
autoencoder-v3/deconv2/BiasAdd«
&autoencoder-v3/deconv2-leaky/LeakyRelu	LeakyRelu'autoencoder-v3/deconv2/BiasAdd:output:0*/
_output_shapes
:€€€€€€€€€  *
alpha%Ќћћ=2(
&autoencoder-v3/deconv2-leaky/LeakyRelu†
autoencoder-v3/deconv1/ShapeShape4autoencoder-v3/deconv2-leaky/LeakyRelu:activations:0*
T0*
_output_shapes
:2
autoencoder-v3/deconv1/ShapeҐ
*autoencoder-v3/deconv1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*autoencoder-v3/deconv1/strided_slice/stack¶
,autoencoder-v3/deconv1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,autoencoder-v3/deconv1/strided_slice/stack_1¶
,autoencoder-v3/deconv1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,autoencoder-v3/deconv1/strided_slice/stack_2м
$autoencoder-v3/deconv1/strided_sliceStridedSlice%autoencoder-v3/deconv1/Shape:output:03autoencoder-v3/deconv1/strided_slice/stack:output:05autoencoder-v3/deconv1/strided_slice/stack_1:output:05autoencoder-v3/deconv1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$autoencoder-v3/deconv1/strided_sliceВ
autoencoder-v3/deconv1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@2 
autoencoder-v3/deconv1/stack/1В
autoencoder-v3/deconv1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@2 
autoencoder-v3/deconv1/stack/2В
autoencoder-v3/deconv1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2 
autoencoder-v3/deconv1/stack/3Ь
autoencoder-v3/deconv1/stackPack-autoencoder-v3/deconv1/strided_slice:output:0'autoencoder-v3/deconv1/stack/1:output:0'autoencoder-v3/deconv1/stack/2:output:0'autoencoder-v3/deconv1/stack/3:output:0*
N*
T0*
_output_shapes
:2
autoencoder-v3/deconv1/stack¶
,autoencoder-v3/deconv1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,autoencoder-v3/deconv1/strided_slice_1/stack™
.autoencoder-v3/deconv1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.autoencoder-v3/deconv1/strided_slice_1/stack_1™
.autoencoder-v3/deconv1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.autoencoder-v3/deconv1/strided_slice_1/stack_2ц
&autoencoder-v3/deconv1/strided_slice_1StridedSlice%autoencoder-v3/deconv1/stack:output:05autoencoder-v3/deconv1/strided_slice_1/stack:output:07autoencoder-v3/deconv1/strided_slice_1/stack_1:output:07autoencoder-v3/deconv1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&autoencoder-v3/deconv1/strided_slice_1ш
6autoencoder-v3/deconv1/conv2d_transpose/ReadVariableOpReadVariableOp?autoencoder_v3_deconv1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype028
6autoencoder-v3/deconv1/conv2d_transpose/ReadVariableOpй
'autoencoder-v3/deconv1/conv2d_transposeConv2DBackpropInput%autoencoder-v3/deconv1/stack:output:0>autoencoder-v3/deconv1/conv2d_transpose/ReadVariableOp:value:04autoencoder-v3/deconv2-leaky/LeakyRelu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€@@*
paddingVALID*
strides
2)
'autoencoder-v3/deconv1/conv2d_transpose—
-autoencoder-v3/deconv1/BiasAdd/ReadVariableOpReadVariableOp6autoencoder_v3_deconv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-autoencoder-v3/deconv1/BiasAdd/ReadVariableOpо
autoencoder-v3/deconv1/BiasAddBiasAdd0autoencoder-v3/deconv1/conv2d_transpose:output:05autoencoder-v3/deconv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@@2 
autoencoder-v3/deconv1/BiasAddЊ
&autoencoder-v3/deconv1-sigmoid/SigmoidSigmoid'autoencoder-v3/deconv1/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€@@2(
&autoencoder-v3/deconv1-sigmoid/SigmoidЖ
IdentityIdentity*autoencoder-v3/deconv1-sigmoid/Sigmoid:y:0*
T0*/
_output_shapes
:€€€€€€€€€@@2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:€€€€€€€€€@@:::::::::::::X T
/
_output_shapes
:€€€€€€€€€@@
!
_user_specified_name	input_2
Щq
≠
!__inference__traced_restore_41812
file_prefix!
assignvariableop_conv1_kernel!
assignvariableop_1_conv1_bias#
assignvariableop_2_conv2_kernel!
assignvariableop_3_conv2_bias#
assignvariableop_4_conv3_kernel!
assignvariableop_5_conv3_bias%
!assignvariableop_6_deconv3_kernel#
assignvariableop_7_deconv3_bias%
!assignvariableop_8_deconv2_kernel#
assignvariableop_9_deconv2_bias&
"assignvariableop_10_deconv1_kernel$
 assignvariableop_11_deconv1_bias!
assignvariableop_12_adam_iter#
assignvariableop_13_adam_beta_1#
assignvariableop_14_adam_beta_2"
assignvariableop_15_adam_decay*
&assignvariableop_16_adam_learning_rate
assignvariableop_17_total
assignvariableop_18_count+
'assignvariableop_19_adam_conv3_kernel_m)
%assignvariableop_20_adam_conv3_bias_m-
)assignvariableop_21_adam_deconv3_kernel_m+
'assignvariableop_22_adam_deconv3_bias_m+
'assignvariableop_23_adam_conv3_kernel_v)
%assignvariableop_24_adam_conv3_bias_v-
)assignvariableop_25_adam_deconv3_kernel_v+
'assignvariableop_26_adam_deconv3_bias_v
identity_28ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_10ҐAssignVariableOp_11ҐAssignVariableOp_12ҐAssignVariableOp_13ҐAssignVariableOp_14ҐAssignVariableOp_15ҐAssignVariableOp_16ҐAssignVariableOp_17ҐAssignVariableOp_18ҐAssignVariableOp_19ҐAssignVariableOp_2ҐAssignVariableOp_20ҐAssignVariableOp_21ҐAssignVariableOp_22ҐAssignVariableOp_23ҐAssignVariableOp_24ҐAssignVariableOp_25ҐAssignVariableOp_26ҐAssignVariableOp_3ҐAssignVariableOp_4ҐAssignVariableOp_5ҐAssignVariableOp_6ҐAssignVariableOp_7ҐAssignVariableOp_8ҐAssignVariableOp_9§
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*∞
value¶B£B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names∆
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*K
valueBB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesЄ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Д
_output_shapesr
p::::::::::::::::::::::::::::**
dtypes 
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

IdentityЬ
AssignVariableOpAssignVariableOpassignvariableop_conv1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1Ґ
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2§
AssignVariableOp_2AssignVariableOpassignvariableop_2_conv2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3Ґ
AssignVariableOp_3AssignVariableOpassignvariableop_3_conv2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4§
AssignVariableOp_4AssignVariableOpassignvariableop_4_conv3_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5Ґ
AssignVariableOp_5AssignVariableOpassignvariableop_5_conv3_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6¶
AssignVariableOp_6AssignVariableOp!assignvariableop_6_deconv3_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7§
AssignVariableOp_7AssignVariableOpassignvariableop_7_deconv3_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8¶
AssignVariableOp_8AssignVariableOp!assignvariableop_8_deconv2_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9§
AssignVariableOp_9AssignVariableOpassignvariableop_9_deconv2_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10™
AssignVariableOp_10AssignVariableOp"assignvariableop_10_deconv1_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11®
AssignVariableOp_11AssignVariableOp assignvariableop_11_deconv1_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_12•
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_iterIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13І
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_beta_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14І
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_beta_2Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15¶
AssignVariableOp_15AssignVariableOpassignvariableop_15_adam_decayIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16Ѓ
AssignVariableOp_16AssignVariableOp&assignvariableop_16_adam_learning_rateIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17°
AssignVariableOp_17AssignVariableOpassignvariableop_17_totalIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18°
AssignVariableOp_18AssignVariableOpassignvariableop_18_countIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19ѓ
AssignVariableOp_19AssignVariableOp'assignvariableop_19_adam_conv3_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20≠
AssignVariableOp_20AssignVariableOp%assignvariableop_20_adam_conv3_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21±
AssignVariableOp_21AssignVariableOp)assignvariableop_21_adam_deconv3_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22ѓ
AssignVariableOp_22AssignVariableOp'assignvariableop_22_adam_deconv3_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23ѓ
AssignVariableOp_23AssignVariableOp'assignvariableop_23_adam_conv3_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24≠
AssignVariableOp_24AssignVariableOp%assignvariableop_24_adam_conv3_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25±
AssignVariableOp_25AssignVariableOp)assignvariableop_25_adam_deconv3_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26ѓ
AssignVariableOp_26AssignVariableOp'assignvariableop_26_adam_deconv3_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_269
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp∞
Identity_27Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_27£
Identity_28IdentityIdentity_27:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_28"#
identity_28Identity_28:output:0*Б
_input_shapesp
n: :::::::::::::::::::::::::::2$
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
AssignVariableOp_26AssignVariableOp_262(
AssignVariableOp_3AssignVariableOp_32(
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
п
b
F__inference_conv3-leaky_layer_call_and_return_conditional_losses_40991

inputs
identityl
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:€€€€€€€€€*
alpha%Ќћћ=2
	LeakyRelus
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Ґ
®
@__inference_conv1_layer_call_and_return_conditional_losses_41510

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€  *
paddingVALID*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€  2	
BiasAddl
IdentityIdentityBiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€  2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€@@:::W S
/
_output_shapes
:€€€€€€€€€@@
 
_user_specified_nameinputs
є
G
+__inference_conv2-leaky_layer_call_fn_41558

inputs
identityѕ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_conv2-leaky_layer_call_and_return_conditional_losses_409522
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:€€€€€€€€€
2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€
:W S
/
_output_shapes
:€€€€€€€€€

 
_user_specified_nameinputs
р	
Ъ
.__inference_autoencoder-v3_layer_call_fn_41471

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identityИҐStatefulPartitionedCallЩ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_autoencoder-v3_layer_call_and_return_conditional_losses_411372
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:€€€€€€€€€@@::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€@@
 
_user_specified_nameinputs
Ґ
®
@__inference_conv3_layer_call_and_return_conditional_losses_41568

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingVALID*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2	
BiasAddl
IdentityIdentityBiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€
:::W S
/
_output_shapes
:€€€€€€€€€

 
_user_specified_nameinputs
¬$
і
B__inference_deconv3_layer_call_and_return_conditional_losses_40772

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identityИD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2в
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2м
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2м
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulP
add/yConst*
_output_shapes
: *
dtype0*
value	B : 2
add/yM
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: 2
addT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
add_1/yConst*
_output_shapes
: *
dtype0*
value	B : 2	
add_1/yU
add_1AddV2	mul_1:z:0add_1/y:output:0*
T0*
_output_shapes
: 2
add_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :
2	
stack/3В
stackPackstrided_slice:output:0add:z:0	add_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2м
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3≥
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:
*
dtype02!
conv2d_transpose/ReadVariableOpс
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
*
paddingVALID*
strides
2
conv2d_transposeМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp§
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
2	
BiasAdd~
IdentityIdentityBiasAdd:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:::i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ч
z
%__inference_conv1_layer_call_fn_41519

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_conv1_layer_call_and_return_conditional_losses_408922
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€  2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€@@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€@@
 
_user_specified_nameinputs
¬$
і
B__inference_deconv2_layer_call_and_return_conditional_losses_40820

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identityИD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2в
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2м
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2м
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulP
add/yConst*
_output_shapes
: *
dtype0*
value	B : 2
add/yM
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: 2
addT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
add_1/yConst*
_output_shapes
: *
dtype0*
value	B : 2	
add_1/yU
add_1AddV2	mul_1:z:0add_1/y:output:0*
T0*
_output_shapes
: 2
add_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3В
stackPackstrided_slice:output:0add:z:0	add_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2м
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3≥
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:
*
dtype02!
conv2d_transpose/ReadVariableOpс
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*
paddingVALID*
strides
2
conv2d_transposeМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp§
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2	
BiasAdd~
IdentityIdentityBiasAdd:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
:::i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€

 
_user_specified_nameinputs
Ж
I
-__inference_deconv2-leaky_layer_call_fn_41607

inputs
identityг
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_deconv2-leaky_layer_call_and_return_conditional_losses_410272
PartitionedCallЖ
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Ы	
Р
#__inference_signature_wrapper_41272
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identityИҐStatefulPartitionedCallя
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@@*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *)
f$R"
 __inference__wrapped_model_407342
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€@@2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:€€€€€€€€€@@::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:€€€€€€€€€@@
!
_user_specified_name	input_2
Ґ
®
@__inference_conv3_layer_call_and_return_conditional_losses_40970

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingVALID*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2	
BiasAddl
IdentityIdentityBiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€
:::W S
/
_output_shapes
:€€€€€€€€€

 
_user_specified_nameinputs
п
b
F__inference_conv1-leaky_layer_call_and_return_conditional_losses_40913

inputs
identityl
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:€€€€€€€€€  *
alpha%Ќћћ=2
	LeakyRelus
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€  2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€  :W S
/
_output_shapes
:€€€€€€€€€  
 
_user_specified_nameinputs
µb
Р
I__inference_autoencoder-v3_layer_call_and_return_conditional_losses_41442

inputs(
$conv1_conv2d_readvariableop_resource)
%conv1_biasadd_readvariableop_resource(
$conv2_conv2d_readvariableop_resource)
%conv2_biasadd_readvariableop_resource(
$conv3_conv2d_readvariableop_resource)
%conv3_biasadd_readvariableop_resource4
0deconv3_conv2d_transpose_readvariableop_resource+
'deconv3_biasadd_readvariableop_resource4
0deconv2_conv2d_transpose_readvariableop_resource+
'deconv2_biasadd_readvariableop_resource4
0deconv1_conv2d_transpose_readvariableop_resource+
'deconv1_biasadd_readvariableop_resource
identityИІ
conv1/Conv2D/ReadVariableOpReadVariableOp$conv1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv1/Conv2D/ReadVariableOpґ
conv1/Conv2DConv2Dinputs#conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€  *
paddingVALID*
strides
2
conv1/Conv2DЮ
conv1/BiasAdd/ReadVariableOpReadVariableOp%conv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv1/BiasAdd/ReadVariableOp†
conv1/BiasAddBiasAddconv1/Conv2D:output:0$conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€  2
conv1/BiasAddФ
conv1-leaky/LeakyRelu	LeakyReluconv1/BiasAdd:output:0*/
_output_shapes
:€€€€€€€€€  *
alpha%Ќћћ=2
conv1-leaky/LeakyReluІ
conv2/Conv2D/ReadVariableOpReadVariableOp$conv2_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02
conv2/Conv2D/ReadVariableOp”
conv2/Conv2DConv2D#conv1-leaky/LeakyRelu:activations:0#conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€
*
paddingVALID*
strides
2
conv2/Conv2DЮ
conv2/BiasAdd/ReadVariableOpReadVariableOp%conv2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
conv2/BiasAdd/ReadVariableOp†
conv2/BiasAddBiasAddconv2/Conv2D:output:0$conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€
2
conv2/BiasAddФ
conv2-leaky/LeakyRelu	LeakyReluconv2/BiasAdd:output:0*/
_output_shapes
:€€€€€€€€€
*
alpha%Ќћћ=2
conv2-leaky/LeakyReluІ
conv3/Conv2D/ReadVariableOpReadVariableOp$conv3_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02
conv3/Conv2D/ReadVariableOp”
conv3/Conv2DConv2D#conv2-leaky/LeakyRelu:activations:0#conv3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingVALID*
strides
2
conv3/Conv2DЮ
conv3/BiasAdd/ReadVariableOpReadVariableOp%conv3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv3/BiasAdd/ReadVariableOp†
conv3/BiasAddBiasAddconv3/Conv2D:output:0$conv3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2
conv3/BiasAddФ
conv3-leaky/LeakyRelu	LeakyReluconv3/BiasAdd:output:0*/
_output_shapes
:€€€€€€€€€*
alpha%Ќћћ=2
conv3-leaky/LeakyReluq
deconv3/ShapeShape#conv3-leaky/LeakyRelu:activations:0*
T0*
_output_shapes
:2
deconv3/ShapeД
deconv3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
deconv3/strided_slice/stackИ
deconv3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
deconv3/strided_slice/stack_1И
deconv3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
deconv3/strided_slice/stack_2Т
deconv3/strided_sliceStridedSlicedeconv3/Shape:output:0$deconv3/strided_slice/stack:output:0&deconv3/strided_slice/stack_1:output:0&deconv3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
deconv3/strided_sliced
deconv3/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
deconv3/stack/1d
deconv3/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
deconv3/stack/2d
deconv3/stack/3Const*
_output_shapes
: *
dtype0*
value	B :
2
deconv3/stack/3¬
deconv3/stackPackdeconv3/strided_slice:output:0deconv3/stack/1:output:0deconv3/stack/2:output:0deconv3/stack/3:output:0*
N*
T0*
_output_shapes
:2
deconv3/stackИ
deconv3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
deconv3/strided_slice_1/stackМ
deconv3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
deconv3/strided_slice_1/stack_1М
deconv3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
deconv3/strided_slice_1/stack_2Ь
deconv3/strided_slice_1StridedSlicedeconv3/stack:output:0&deconv3/strided_slice_1/stack:output:0(deconv3/strided_slice_1/stack_1:output:0(deconv3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
deconv3/strided_slice_1Ћ
'deconv3/conv2d_transpose/ReadVariableOpReadVariableOp0deconv3_conv2d_transpose_readvariableop_resource*&
_output_shapes
:
*
dtype02)
'deconv3/conv2d_transpose/ReadVariableOpЬ
deconv3/conv2d_transposeConv2DBackpropInputdeconv3/stack:output:0/deconv3/conv2d_transpose/ReadVariableOp:value:0#conv3-leaky/LeakyRelu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€
*
paddingVALID*
strides
2
deconv3/conv2d_transpose§
deconv3/BiasAdd/ReadVariableOpReadVariableOp'deconv3_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
deconv3/BiasAdd/ReadVariableOp≤
deconv3/BiasAddBiasAdd!deconv3/conv2d_transpose:output:0&deconv3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€
2
deconv3/BiasAddЪ
deconv3-leaky/LeakyRelu	LeakyReludeconv3/BiasAdd:output:0*/
_output_shapes
:€€€€€€€€€
*
alpha%Ќћћ=2
deconv3-leaky/LeakyRelus
deconv2/ShapeShape%deconv3-leaky/LeakyRelu:activations:0*
T0*
_output_shapes
:2
deconv2/ShapeД
deconv2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
deconv2/strided_slice/stackИ
deconv2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
deconv2/strided_slice/stack_1И
deconv2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
deconv2/strided_slice/stack_2Т
deconv2/strided_sliceStridedSlicedeconv2/Shape:output:0$deconv2/strided_slice/stack:output:0&deconv2/strided_slice/stack_1:output:0&deconv2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
deconv2/strided_sliced
deconv2/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 2
deconv2/stack/1d
deconv2/stack/2Const*
_output_shapes
: *
dtype0*
value	B : 2
deconv2/stack/2d
deconv2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
deconv2/stack/3¬
deconv2/stackPackdeconv2/strided_slice:output:0deconv2/stack/1:output:0deconv2/stack/2:output:0deconv2/stack/3:output:0*
N*
T0*
_output_shapes
:2
deconv2/stackИ
deconv2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
deconv2/strided_slice_1/stackМ
deconv2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
deconv2/strided_slice_1/stack_1М
deconv2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
deconv2/strided_slice_1/stack_2Ь
deconv2/strided_slice_1StridedSlicedeconv2/stack:output:0&deconv2/strided_slice_1/stack:output:0(deconv2/strided_slice_1/stack_1:output:0(deconv2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
deconv2/strided_slice_1Ћ
'deconv2/conv2d_transpose/ReadVariableOpReadVariableOp0deconv2_conv2d_transpose_readvariableop_resource*&
_output_shapes
:
*
dtype02)
'deconv2/conv2d_transpose/ReadVariableOpЮ
deconv2/conv2d_transposeConv2DBackpropInputdeconv2/stack:output:0/deconv2/conv2d_transpose/ReadVariableOp:value:0%deconv3-leaky/LeakyRelu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€  *
paddingVALID*
strides
2
deconv2/conv2d_transpose§
deconv2/BiasAdd/ReadVariableOpReadVariableOp'deconv2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
deconv2/BiasAdd/ReadVariableOp≤
deconv2/BiasAddBiasAdd!deconv2/conv2d_transpose:output:0&deconv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€  2
deconv2/BiasAddЪ
deconv2-leaky/LeakyRelu	LeakyReludeconv2/BiasAdd:output:0*/
_output_shapes
:€€€€€€€€€  *
alpha%Ќћћ=2
deconv2-leaky/LeakyRelus
deconv1/ShapeShape%deconv2-leaky/LeakyRelu:activations:0*
T0*
_output_shapes
:2
deconv1/ShapeД
deconv1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
deconv1/strided_slice/stackИ
deconv1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
deconv1/strided_slice/stack_1И
deconv1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
deconv1/strided_slice/stack_2Т
deconv1/strided_sliceStridedSlicedeconv1/Shape:output:0$deconv1/strided_slice/stack:output:0&deconv1/strided_slice/stack_1:output:0&deconv1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
deconv1/strided_sliced
deconv1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@2
deconv1/stack/1d
deconv1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@2
deconv1/stack/2d
deconv1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
deconv1/stack/3¬
deconv1/stackPackdeconv1/strided_slice:output:0deconv1/stack/1:output:0deconv1/stack/2:output:0deconv1/stack/3:output:0*
N*
T0*
_output_shapes
:2
deconv1/stackИ
deconv1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
deconv1/strided_slice_1/stackМ
deconv1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
deconv1/strided_slice_1/stack_1М
deconv1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
deconv1/strided_slice_1/stack_2Ь
deconv1/strided_slice_1StridedSlicedeconv1/stack:output:0&deconv1/strided_slice_1/stack:output:0(deconv1/strided_slice_1/stack_1:output:0(deconv1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
deconv1/strided_slice_1Ћ
'deconv1/conv2d_transpose/ReadVariableOpReadVariableOp0deconv1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02)
'deconv1/conv2d_transpose/ReadVariableOpЮ
deconv1/conv2d_transposeConv2DBackpropInputdeconv1/stack:output:0/deconv1/conv2d_transpose/ReadVariableOp:value:0%deconv2-leaky/LeakyRelu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€@@*
paddingVALID*
strides
2
deconv1/conv2d_transpose§
deconv1/BiasAdd/ReadVariableOpReadVariableOp'deconv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
deconv1/BiasAdd/ReadVariableOp≤
deconv1/BiasAddBiasAdd!deconv1/conv2d_transpose:output:0&deconv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@@2
deconv1/BiasAddС
deconv1-sigmoid/SigmoidSigmoiddeconv1/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€@@2
deconv1-sigmoid/Sigmoidw
IdentityIdentitydeconv1-sigmoid/Sigmoid:y:0*
T0*/
_output_shapes
:€€€€€€€€€@@2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:€€€€€€€€€@@:::::::::::::W S
/
_output_shapes
:€€€€€€€€€@@
 
_user_specified_nameinputs
ч
z
%__inference_conv2_layer_call_fn_41548

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_conv2_layer_call_and_return_conditional_losses_409312
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€  ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€  
 
_user_specified_nameinputs
Ґ
f
J__inference_deconv1-sigmoid_layer_call_and_return_conditional_losses_41612

inputs
identityq
SigmoidSigmoidinputs*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2	
Sigmoidy
IdentityIdentitySigmoid:y:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
є
G
+__inference_conv3-leaky_layer_call_fn_41587

inputs
identityѕ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_conv3-leaky_layer_call_and_return_conditional_losses_409912
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
™3
Ж
I__inference_autoencoder-v3_layer_call_and_return_conditional_losses_41137

inputs
conv1_41100
conv1_41102
conv2_41106
conv2_41108
conv3_41112
conv3_41114
deconv3_41118
deconv3_41120
deconv2_41124
deconv2_41126
deconv1_41130
deconv1_41132
identityИҐconv1/StatefulPartitionedCallҐconv2/StatefulPartitionedCallҐconv3/StatefulPartitionedCallҐdeconv1/StatefulPartitionedCallҐdeconv2/StatefulPartitionedCallҐdeconv3/StatefulPartitionedCallН
conv1/StatefulPartitionedCallStatefulPartitionedCallinputsconv1_41100conv1_41102*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_conv1_layer_call_and_return_conditional_losses_408922
conv1/StatefulPartitionedCallЗ
conv1-leaky/PartitionedCallPartitionedCall&conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€  * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_conv1-leaky_layer_call_and_return_conditional_losses_409132
conv1-leaky/PartitionedCallЂ
conv2/StatefulPartitionedCallStatefulPartitionedCall$conv1-leaky/PartitionedCall:output:0conv2_41106conv2_41108*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_conv2_layer_call_and_return_conditional_losses_409312
conv2/StatefulPartitionedCallЗ
conv2-leaky/PartitionedCallPartitionedCall&conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_conv2-leaky_layer_call_and_return_conditional_losses_409522
conv2-leaky/PartitionedCallЂ
conv3/StatefulPartitionedCallStatefulPartitionedCall$conv2-leaky/PartitionedCall:output:0conv3_41112conv3_41114*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_conv3_layer_call_and_return_conditional_losses_409702
conv3/StatefulPartitionedCallЗ
conv3-leaky/PartitionedCallPartitionedCall&conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_conv3-leaky_layer_call_and_return_conditional_losses_409912
conv3-leaky/PartitionedCall«
deconv3/StatefulPartitionedCallStatefulPartitionedCall$conv3-leaky/PartitionedCall:output:0deconv3_41118deconv3_41120*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_deconv3_layer_call_and_return_conditional_losses_407722!
deconv3/StatefulPartitionedCall°
deconv3-leaky/PartitionedCallPartitionedCall(deconv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_deconv3-leaky_layer_call_and_return_conditional_losses_410092
deconv3-leaky/PartitionedCall…
deconv2/StatefulPartitionedCallStatefulPartitionedCall&deconv3-leaky/PartitionedCall:output:0deconv2_41124deconv2_41126*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_deconv2_layer_call_and_return_conditional_losses_408202!
deconv2/StatefulPartitionedCall°
deconv2-leaky/PartitionedCallPartitionedCall(deconv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_deconv2-leaky_layer_call_and_return_conditional_losses_410272
deconv2-leaky/PartitionedCall…
deconv1/StatefulPartitionedCallStatefulPartitionedCall&deconv2-leaky/PartitionedCall:output:0deconv1_41130deconv1_41132*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_deconv1_layer_call_and_return_conditional_losses_408682!
deconv1/StatefulPartitionedCallІ
deconv1-sigmoid/PartitionedCallPartitionedCall(deconv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_deconv1-sigmoid_layer_call_and_return_conditional_losses_410452!
deconv1-sigmoid/PartitionedCall№
IdentityIdentity(deconv1-sigmoid/PartitionedCall:output:0^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^conv3/StatefulPartitionedCall ^deconv1/StatefulPartitionedCall ^deconv2/StatefulPartitionedCall ^deconv3/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:€€€€€€€€€@@::::::::::::2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2>
conv3/StatefulPartitionedCallconv3/StatefulPartitionedCall2B
deconv1/StatefulPartitionedCalldeconv1/StatefulPartitionedCall2B
deconv2/StatefulPartitionedCalldeconv2/StatefulPartitionedCall2B
deconv3/StatefulPartitionedCalldeconv3/StatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€@@
 
_user_specified_nameinputs
п
b
F__inference_conv1-leaky_layer_call_and_return_conditional_losses_41524

inputs
identityl
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:€€€€€€€€€  *
alpha%Ќћћ=2
	LeakyRelus
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€  2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€  :W S
/
_output_shapes
:€€€€€€€€€  
 
_user_specified_nameinputs
√
|
'__inference_deconv3_layer_call_fn_40782

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallП
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_deconv3_layer_call_and_return_conditional_losses_407722
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Ї
d
H__inference_deconv3-leaky_layer_call_and_return_conditional_losses_41592

inputs
identity~
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
*
alpha%Ќћћ=2
	LeakyReluЕ
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€

 
_user_specified_nameinputs
µb
Р
I__inference_autoencoder-v3_layer_call_and_return_conditional_losses_41357

inputs(
$conv1_conv2d_readvariableop_resource)
%conv1_biasadd_readvariableop_resource(
$conv2_conv2d_readvariableop_resource)
%conv2_biasadd_readvariableop_resource(
$conv3_conv2d_readvariableop_resource)
%conv3_biasadd_readvariableop_resource4
0deconv3_conv2d_transpose_readvariableop_resource+
'deconv3_biasadd_readvariableop_resource4
0deconv2_conv2d_transpose_readvariableop_resource+
'deconv2_biasadd_readvariableop_resource4
0deconv1_conv2d_transpose_readvariableop_resource+
'deconv1_biasadd_readvariableop_resource
identityИІ
conv1/Conv2D/ReadVariableOpReadVariableOp$conv1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv1/Conv2D/ReadVariableOpґ
conv1/Conv2DConv2Dinputs#conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€  *
paddingVALID*
strides
2
conv1/Conv2DЮ
conv1/BiasAdd/ReadVariableOpReadVariableOp%conv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv1/BiasAdd/ReadVariableOp†
conv1/BiasAddBiasAddconv1/Conv2D:output:0$conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€  2
conv1/BiasAddФ
conv1-leaky/LeakyRelu	LeakyReluconv1/BiasAdd:output:0*/
_output_shapes
:€€€€€€€€€  *
alpha%Ќћћ=2
conv1-leaky/LeakyReluІ
conv2/Conv2D/ReadVariableOpReadVariableOp$conv2_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02
conv2/Conv2D/ReadVariableOp”
conv2/Conv2DConv2D#conv1-leaky/LeakyRelu:activations:0#conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€
*
paddingVALID*
strides
2
conv2/Conv2DЮ
conv2/BiasAdd/ReadVariableOpReadVariableOp%conv2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
conv2/BiasAdd/ReadVariableOp†
conv2/BiasAddBiasAddconv2/Conv2D:output:0$conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€
2
conv2/BiasAddФ
conv2-leaky/LeakyRelu	LeakyReluconv2/BiasAdd:output:0*/
_output_shapes
:€€€€€€€€€
*
alpha%Ќћћ=2
conv2-leaky/LeakyReluІ
conv3/Conv2D/ReadVariableOpReadVariableOp$conv3_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02
conv3/Conv2D/ReadVariableOp”
conv3/Conv2DConv2D#conv2-leaky/LeakyRelu:activations:0#conv3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingVALID*
strides
2
conv3/Conv2DЮ
conv3/BiasAdd/ReadVariableOpReadVariableOp%conv3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv3/BiasAdd/ReadVariableOp†
conv3/BiasAddBiasAddconv3/Conv2D:output:0$conv3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2
conv3/BiasAddФ
conv3-leaky/LeakyRelu	LeakyReluconv3/BiasAdd:output:0*/
_output_shapes
:€€€€€€€€€*
alpha%Ќћћ=2
conv3-leaky/LeakyReluq
deconv3/ShapeShape#conv3-leaky/LeakyRelu:activations:0*
T0*
_output_shapes
:2
deconv3/ShapeД
deconv3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
deconv3/strided_slice/stackИ
deconv3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
deconv3/strided_slice/stack_1И
deconv3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
deconv3/strided_slice/stack_2Т
deconv3/strided_sliceStridedSlicedeconv3/Shape:output:0$deconv3/strided_slice/stack:output:0&deconv3/strided_slice/stack_1:output:0&deconv3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
deconv3/strided_sliced
deconv3/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
deconv3/stack/1d
deconv3/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
deconv3/stack/2d
deconv3/stack/3Const*
_output_shapes
: *
dtype0*
value	B :
2
deconv3/stack/3¬
deconv3/stackPackdeconv3/strided_slice:output:0deconv3/stack/1:output:0deconv3/stack/2:output:0deconv3/stack/3:output:0*
N*
T0*
_output_shapes
:2
deconv3/stackИ
deconv3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
deconv3/strided_slice_1/stackМ
deconv3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
deconv3/strided_slice_1/stack_1М
deconv3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
deconv3/strided_slice_1/stack_2Ь
deconv3/strided_slice_1StridedSlicedeconv3/stack:output:0&deconv3/strided_slice_1/stack:output:0(deconv3/strided_slice_1/stack_1:output:0(deconv3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
deconv3/strided_slice_1Ћ
'deconv3/conv2d_transpose/ReadVariableOpReadVariableOp0deconv3_conv2d_transpose_readvariableop_resource*&
_output_shapes
:
*
dtype02)
'deconv3/conv2d_transpose/ReadVariableOpЬ
deconv3/conv2d_transposeConv2DBackpropInputdeconv3/stack:output:0/deconv3/conv2d_transpose/ReadVariableOp:value:0#conv3-leaky/LeakyRelu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€
*
paddingVALID*
strides
2
deconv3/conv2d_transpose§
deconv3/BiasAdd/ReadVariableOpReadVariableOp'deconv3_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
deconv3/BiasAdd/ReadVariableOp≤
deconv3/BiasAddBiasAdd!deconv3/conv2d_transpose:output:0&deconv3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€
2
deconv3/BiasAddЪ
deconv3-leaky/LeakyRelu	LeakyReludeconv3/BiasAdd:output:0*/
_output_shapes
:€€€€€€€€€
*
alpha%Ќћћ=2
deconv3-leaky/LeakyRelus
deconv2/ShapeShape%deconv3-leaky/LeakyRelu:activations:0*
T0*
_output_shapes
:2
deconv2/ShapeД
deconv2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
deconv2/strided_slice/stackИ
deconv2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
deconv2/strided_slice/stack_1И
deconv2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
deconv2/strided_slice/stack_2Т
deconv2/strided_sliceStridedSlicedeconv2/Shape:output:0$deconv2/strided_slice/stack:output:0&deconv2/strided_slice/stack_1:output:0&deconv2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
deconv2/strided_sliced
deconv2/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 2
deconv2/stack/1d
deconv2/stack/2Const*
_output_shapes
: *
dtype0*
value	B : 2
deconv2/stack/2d
deconv2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
deconv2/stack/3¬
deconv2/stackPackdeconv2/strided_slice:output:0deconv2/stack/1:output:0deconv2/stack/2:output:0deconv2/stack/3:output:0*
N*
T0*
_output_shapes
:2
deconv2/stackИ
deconv2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
deconv2/strided_slice_1/stackМ
deconv2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
deconv2/strided_slice_1/stack_1М
deconv2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
deconv2/strided_slice_1/stack_2Ь
deconv2/strided_slice_1StridedSlicedeconv2/stack:output:0&deconv2/strided_slice_1/stack:output:0(deconv2/strided_slice_1/stack_1:output:0(deconv2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
deconv2/strided_slice_1Ћ
'deconv2/conv2d_transpose/ReadVariableOpReadVariableOp0deconv2_conv2d_transpose_readvariableop_resource*&
_output_shapes
:
*
dtype02)
'deconv2/conv2d_transpose/ReadVariableOpЮ
deconv2/conv2d_transposeConv2DBackpropInputdeconv2/stack:output:0/deconv2/conv2d_transpose/ReadVariableOp:value:0%deconv3-leaky/LeakyRelu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€  *
paddingVALID*
strides
2
deconv2/conv2d_transpose§
deconv2/BiasAdd/ReadVariableOpReadVariableOp'deconv2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
deconv2/BiasAdd/ReadVariableOp≤
deconv2/BiasAddBiasAdd!deconv2/conv2d_transpose:output:0&deconv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€  2
deconv2/BiasAddЪ
deconv2-leaky/LeakyRelu	LeakyReludeconv2/BiasAdd:output:0*/
_output_shapes
:€€€€€€€€€  *
alpha%Ќћћ=2
deconv2-leaky/LeakyRelus
deconv1/ShapeShape%deconv2-leaky/LeakyRelu:activations:0*
T0*
_output_shapes
:2
deconv1/ShapeД
deconv1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
deconv1/strided_slice/stackИ
deconv1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
deconv1/strided_slice/stack_1И
deconv1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
deconv1/strided_slice/stack_2Т
deconv1/strided_sliceStridedSlicedeconv1/Shape:output:0$deconv1/strided_slice/stack:output:0&deconv1/strided_slice/stack_1:output:0&deconv1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
deconv1/strided_sliced
deconv1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@2
deconv1/stack/1d
deconv1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@2
deconv1/stack/2d
deconv1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
deconv1/stack/3¬
deconv1/stackPackdeconv1/strided_slice:output:0deconv1/stack/1:output:0deconv1/stack/2:output:0deconv1/stack/3:output:0*
N*
T0*
_output_shapes
:2
deconv1/stackИ
deconv1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
deconv1/strided_slice_1/stackМ
deconv1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
deconv1/strided_slice_1/stack_1М
deconv1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
deconv1/strided_slice_1/stack_2Ь
deconv1/strided_slice_1StridedSlicedeconv1/stack:output:0&deconv1/strided_slice_1/stack:output:0(deconv1/strided_slice_1/stack_1:output:0(deconv1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
deconv1/strided_slice_1Ћ
'deconv1/conv2d_transpose/ReadVariableOpReadVariableOp0deconv1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02)
'deconv1/conv2d_transpose/ReadVariableOpЮ
deconv1/conv2d_transposeConv2DBackpropInputdeconv1/stack:output:0/deconv1/conv2d_transpose/ReadVariableOp:value:0%deconv2-leaky/LeakyRelu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€@@*
paddingVALID*
strides
2
deconv1/conv2d_transpose§
deconv1/BiasAdd/ReadVariableOpReadVariableOp'deconv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
deconv1/BiasAdd/ReadVariableOp≤
deconv1/BiasAddBiasAdd!deconv1/conv2d_transpose:output:0&deconv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@@2
deconv1/BiasAddС
deconv1-sigmoid/SigmoidSigmoiddeconv1/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€@@2
deconv1-sigmoid/Sigmoidw
IdentityIdentitydeconv1-sigmoid/Sigmoid:y:0*
T0*/
_output_shapes
:€€€€€€€€€@@2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:€€€€€€€€€@@:::::::::::::W S
/
_output_shapes
:€€€€€€€€€@@
 
_user_specified_nameinputs
≠3
З
I__inference_autoencoder-v3_layer_call_and_return_conditional_losses_41054
input_2
conv1_40903
conv1_40905
conv2_40942
conv2_40944
conv3_40981
conv3_40983
deconv3_40999
deconv3_41001
deconv2_41017
deconv2_41019
deconv1_41035
deconv1_41037
identityИҐconv1/StatefulPartitionedCallҐconv2/StatefulPartitionedCallҐconv3/StatefulPartitionedCallҐdeconv1/StatefulPartitionedCallҐdeconv2/StatefulPartitionedCallҐdeconv3/StatefulPartitionedCallО
conv1/StatefulPartitionedCallStatefulPartitionedCallinput_2conv1_40903conv1_40905*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_conv1_layer_call_and_return_conditional_losses_408922
conv1/StatefulPartitionedCallЗ
conv1-leaky/PartitionedCallPartitionedCall&conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€  * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_conv1-leaky_layer_call_and_return_conditional_losses_409132
conv1-leaky/PartitionedCallЂ
conv2/StatefulPartitionedCallStatefulPartitionedCall$conv1-leaky/PartitionedCall:output:0conv2_40942conv2_40944*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_conv2_layer_call_and_return_conditional_losses_409312
conv2/StatefulPartitionedCallЗ
conv2-leaky/PartitionedCallPartitionedCall&conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_conv2-leaky_layer_call_and_return_conditional_losses_409522
conv2-leaky/PartitionedCallЂ
conv3/StatefulPartitionedCallStatefulPartitionedCall$conv2-leaky/PartitionedCall:output:0conv3_40981conv3_40983*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_conv3_layer_call_and_return_conditional_losses_409702
conv3/StatefulPartitionedCallЗ
conv3-leaky/PartitionedCallPartitionedCall&conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_conv3-leaky_layer_call_and_return_conditional_losses_409912
conv3-leaky/PartitionedCall«
deconv3/StatefulPartitionedCallStatefulPartitionedCall$conv3-leaky/PartitionedCall:output:0deconv3_40999deconv3_41001*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_deconv3_layer_call_and_return_conditional_losses_407722!
deconv3/StatefulPartitionedCall°
deconv3-leaky/PartitionedCallPartitionedCall(deconv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_deconv3-leaky_layer_call_and_return_conditional_losses_410092
deconv3-leaky/PartitionedCall…
deconv2/StatefulPartitionedCallStatefulPartitionedCall&deconv3-leaky/PartitionedCall:output:0deconv2_41017deconv2_41019*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_deconv2_layer_call_and_return_conditional_losses_408202!
deconv2/StatefulPartitionedCall°
deconv2-leaky/PartitionedCallPartitionedCall(deconv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_deconv2-leaky_layer_call_and_return_conditional_losses_410272
deconv2-leaky/PartitionedCall…
deconv1/StatefulPartitionedCallStatefulPartitionedCall&deconv2-leaky/PartitionedCall:output:0deconv1_41035deconv1_41037*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_deconv1_layer_call_and_return_conditional_losses_408682!
deconv1/StatefulPartitionedCallІ
deconv1-sigmoid/PartitionedCallPartitionedCall(deconv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_deconv1-sigmoid_layer_call_and_return_conditional_losses_410452!
deconv1-sigmoid/PartitionedCall№
IdentityIdentity(deconv1-sigmoid/PartitionedCall:output:0^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^conv3/StatefulPartitionedCall ^deconv1/StatefulPartitionedCall ^deconv2/StatefulPartitionedCall ^deconv3/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:€€€€€€€€€@@::::::::::::2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2>
conv3/StatefulPartitionedCallconv3/StatefulPartitionedCall2B
deconv1/StatefulPartitionedCalldeconv1/StatefulPartitionedCall2B
deconv2/StatefulPartitionedCalldeconv2/StatefulPartitionedCall2B
deconv3/StatefulPartitionedCalldeconv3/StatefulPartitionedCall:X T
/
_output_shapes
:€€€€€€€€€@@
!
_user_specified_name	input_2
√
|
'__inference_deconv2_layer_call_fn_40830

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallП
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_deconv2_layer_call_and_return_conditional_losses_408202
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€

 
_user_specified_nameinputs
Ї
d
H__inference_deconv3-leaky_layer_call_and_return_conditional_losses_41009

inputs
identity~
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
*
alpha%Ќћћ=2
	LeakyReluЕ
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€

 
_user_specified_nameinputs
Ї
d
H__inference_deconv2-leaky_layer_call_and_return_conditional_losses_41027

inputs
identity~
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*
alpha%Ќћћ=2
	LeakyReluЕ
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Ж
I
-__inference_deconv3-leaky_layer_call_fn_41597

inputs
identityг
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_deconv3-leaky_layer_call_and_return_conditional_losses_410092
PartitionedCallЖ
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€

 
_user_specified_nameinputs
п
b
F__inference_conv2-leaky_layer_call_and_return_conditional_losses_41553

inputs
identityl
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:€€€€€€€€€
*
alpha%Ќћћ=2
	LeakyRelus
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€
2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€
:W S
/
_output_shapes
:€€€€€€€€€

 
_user_specified_nameinputs
Я=
”

__inference__traced_save_41721
file_prefix+
'savev2_conv1_kernel_read_readvariableop)
%savev2_conv1_bias_read_readvariableop+
'savev2_conv2_kernel_read_readvariableop)
%savev2_conv2_bias_read_readvariableop+
'savev2_conv3_kernel_read_readvariableop)
%savev2_conv3_bias_read_readvariableop-
)savev2_deconv3_kernel_read_readvariableop+
'savev2_deconv3_bias_read_readvariableop-
)savev2_deconv2_kernel_read_readvariableop+
'savev2_deconv2_bias_read_readvariableop-
)savev2_deconv1_kernel_read_readvariableop+
'savev2_deconv1_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop2
.savev2_adam_conv3_kernel_m_read_readvariableop0
,savev2_adam_conv3_bias_m_read_readvariableop4
0savev2_adam_deconv3_kernel_m_read_readvariableop2
.savev2_adam_deconv3_bias_m_read_readvariableop2
.savev2_adam_conv3_kernel_v_read_readvariableop0
,savev2_adam_conv3_bias_v_read_readvariableop4
0savev2_adam_deconv3_kernel_v_read_readvariableop2
.savev2_adam_deconv3_bias_v_read_readvariableop
savev2_const

identity_1ИҐMergeV2CheckpointsП
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
ConstН
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_78a0018e60f7426bbb6b2ad5f55fc41e/part2	
Const_1Л
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
ShardedFilename/shard¶
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameЮ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*∞
value¶B£B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesј
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*K
valueBB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices—

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_conv1_kernel_read_readvariableop%savev2_conv1_bias_read_readvariableop'savev2_conv2_kernel_read_readvariableop%savev2_conv2_bias_read_readvariableop'savev2_conv3_kernel_read_readvariableop%savev2_conv3_bias_read_readvariableop)savev2_deconv3_kernel_read_readvariableop'savev2_deconv3_bias_read_readvariableop)savev2_deconv2_kernel_read_readvariableop'savev2_deconv2_bias_read_readvariableop)savev2_deconv1_kernel_read_readvariableop'savev2_deconv1_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop.savev2_adam_conv3_kernel_m_read_readvariableop,savev2_adam_conv3_bias_m_read_readvariableop0savev2_adam_deconv3_kernel_m_read_readvariableop.savev2_adam_deconv3_bias_m_read_readvariableop.savev2_adam_conv3_kernel_v_read_readvariableop,savev2_adam_conv3_bias_v_read_readvariableop0savev2_adam_deconv3_kernel_v_read_readvariableop.savev2_adam_deconv3_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 **
dtypes 
2	2
SaveV2Ї
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes°
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*Ч
_input_shapesЕ
В: :::
:
:
::
:
:
:::: : : : : : : :
::
:
:
::
:
: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:
: 

_output_shapes
:
:,(
&
_output_shapes
:
: 

_output_shapes
::,(
&
_output_shapes
:
: 

_output_shapes
:
:,	(
&
_output_shapes
:
: 


_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:
: 

_output_shapes
::,(
&
_output_shapes
:
: 

_output_shapes
:
:,(
&
_output_shapes
:
: 

_output_shapes
::,(
&
_output_shapes
:
: 

_output_shapes
:
:

_output_shapes
: 
√
|
'__inference_deconv1_layer_call_fn_40878

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallП
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_deconv1_layer_call_and_return_conditional_losses_408682
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
п
b
F__inference_conv3-leaky_layer_call_and_return_conditional_losses_41582

inputs
identityl
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:€€€€€€€€€*
alpha%Ќћћ=2
	LeakyRelus
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Ї
d
H__inference_deconv2-leaky_layer_call_and_return_conditional_losses_41602

inputs
identity~
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*
alpha%Ќћћ=2
	LeakyReluЕ
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
р	
Ъ
.__inference_autoencoder-v3_layer_call_fn_41500

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identityИҐStatefulPartitionedCallЩ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_autoencoder-v3_layer_call_and_return_conditional_losses_412062
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:€€€€€€€€€@@::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€@@
 
_user_specified_nameinputs
Ґ
f
J__inference_deconv1-sigmoid_layer_call_and_return_conditional_losses_41045

inputs
identityq
SigmoidSigmoidinputs*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2	
Sigmoidy
IdentityIdentitySigmoid:y:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs"ЄL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*¬
serving_defaultЃ
C
input_28
serving_default_input_2:0€€€€€€€€€@@K
deconv1-sigmoid8
StatefulPartitionedCall:0€€€€€€€€€@@tensorflow/serving/predict:фв
и\
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer-9
layer_with_weights-5
layer-10
layer-11
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
¶_default_save_signature
+І&call_and_return_all_conditional_losses
®__call__"аX
_tf_keras_sequentialЅX{"class_name": "Sequential", "name": "autoencoder-v3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "autoencoder-v3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}}, {"class_name": "Conv2D", "config": {"name": "conv1", "trainable": false, "dtype": "float32", "filters": 4, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "conv1-leaky", "trainable": true, "dtype": "float32", "alpha": 0.10000000149011612}}, {"class_name": "Conv2D", "config": {"name": "conv2", "trainable": false, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "conv2-leaky", "trainable": true, "dtype": "float32", "alpha": 0.10000000149011612}}, {"class_name": "Conv2D", "config": {"name": "conv3", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "conv3-leaky", "trainable": true, "dtype": "float32", "alpha": 0.10000000149011612}}, {"class_name": "Conv2DTranspose", "config": {"name": "deconv3", "trainable": true, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "LeakyReLU", "config": {"name": "deconv3-leaky", "trainable": true, "dtype": "float32", "alpha": 0.10000000149011612}}, {"class_name": "Conv2DTranspose", "config": {"name": "deconv2", "trainable": false, "dtype": "float32", "filters": 4, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "LeakyReLU", "config": {"name": "deconv2-leaky", "trainable": true, "dtype": "float32", "alpha": 0.10000000149011612}}, {"class_name": "Conv2DTranspose", "config": {"name": "deconv1", "trainable": false, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "Activation", "config": {"name": "deconv1-sigmoid", "trainable": true, "dtype": "float32", "activation": "sigmoid"}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 3]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "autoencoder-v3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}}, {"class_name": "Conv2D", "config": {"name": "conv1", "trainable": false, "dtype": "float32", "filters": 4, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "conv1-leaky", "trainable": true, "dtype": "float32", "alpha": 0.10000000149011612}}, {"class_name": "Conv2D", "config": {"name": "conv2", "trainable": false, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "conv2-leaky", "trainable": true, "dtype": "float32", "alpha": 0.10000000149011612}}, {"class_name": "Conv2D", "config": {"name": "conv3", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "conv3-leaky", "trainable": true, "dtype": "float32", "alpha": 0.10000000149011612}}, {"class_name": "Conv2DTranspose", "config": {"name": "deconv3", "trainable": true, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "LeakyReLU", "config": {"name": "deconv3-leaky", "trainable": true, "dtype": "float32", "alpha": 0.10000000149011612}}, {"class_name": "Conv2DTranspose", "config": {"name": "deconv2", "trainable": false, "dtype": "float32", "filters": 4, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "LeakyReLU", "config": {"name": "deconv2-leaky", "trainable": true, "dtype": "float32", "alpha": 0.10000000149011612}}, {"class_name": "Conv2DTranspose", "config": {"name": "deconv1", "trainable": false, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "Activation", "config": {"name": "deconv1-sigmoid", "trainable": true, "dtype": "float32", "activation": "sigmoid"}}]}}, "training_config": {"loss": {"class_name": "MeanSquaredError", "config": {"reduction": "auto", "name": "mean_squared_error"}}, "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.009999999776482582, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
Ф


kernel
bias
#_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api
+©&call_and_return_all_conditional_losses
™__call__"»
_tf_keras_layerЃ{"class_name": "Conv2D", "name": "conv1", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1", "trainable": false, "dtype": "float32", "filters": 4, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 3]}}
№
regularization_losses
	variables
trainable_variables
	keras_api
+Ђ&call_and_return_all_conditional_losses
ђ__call__"Ћ
_tf_keras_layer±{"class_name": "LeakyReLU", "name": "conv1-leaky", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1-leaky", "trainable": true, "dtype": "float32", "alpha": 0.10000000149011612}}
Ц


kernel
bias
# _self_saveable_object_factories
!regularization_losses
"	variables
#trainable_variables
$	keras_api
+≠&call_and_return_all_conditional_losses
Ѓ__call__" 
_tf_keras_layer∞{"class_name": "Conv2D", "name": "conv2", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2", "trainable": false, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 4}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 4]}}
№
%regularization_losses
&	variables
'trainable_variables
(	keras_api
+ѓ&call_and_return_all_conditional_losses
∞__call__"Ћ
_tf_keras_layer±{"class_name": "LeakyReLU", "name": "conv2-leaky", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2-leaky", "trainable": true, "dtype": "float32", "alpha": 0.10000000149011612}}
с	

)kernel
*bias
+regularization_losses
,	variables
-trainable_variables
.	keras_api
+±&call_and_return_all_conditional_losses
≤__call__" 
_tf_keras_layer∞{"class_name": "Conv2D", "name": "conv3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv3", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 10}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 10]}}
№
/regularization_losses
0	variables
1trainable_variables
2	keras_api
+≥&call_and_return_all_conditional_losses
і__call__"Ћ
_tf_keras_layer±{"class_name": "LeakyReLU", "name": "conv3-leaky", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv3-leaky", "trainable": true, "dtype": "float32", "alpha": 0.10000000149011612}}
Ф


3kernel
4bias
5regularization_losses
6	variables
7trainable_variables
8	keras_api
+µ&call_and_return_all_conditional_losses
ґ__call__"н
_tf_keras_layer”{"class_name": "Conv2DTranspose", "name": "deconv3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "deconv3", "trainable": true, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 20}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 8, 20]}}
а
9regularization_losses
:	variables
;trainable_variables
<	keras_api
+Ј&call_and_return_all_conditional_losses
Є__call__"ѕ
_tf_keras_layerµ{"class_name": "LeakyReLU", "name": "deconv3-leaky", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "deconv3-leaky", "trainable": true, "dtype": "float32", "alpha": 0.10000000149011612}}
Љ


=kernel
>bias
#?_self_saveable_object_factories
@regularization_losses
A	variables
Btrainable_variables
C	keras_api
+є&call_and_return_all_conditional_losses
Ї__call__"р
_tf_keras_layer÷{"class_name": "Conv2DTranspose", "name": "deconv2", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "deconv2", "trainable": false, "dtype": "float32", "filters": 4, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 10}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 10]}}
а
Dregularization_losses
E	variables
Ftrainable_variables
G	keras_api
+ї&call_and_return_all_conditional_losses
Љ__call__"ѕ
_tf_keras_layerµ{"class_name": "LeakyReLU", "name": "deconv2-leaky", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "deconv2-leaky", "trainable": true, "dtype": "float32", "alpha": 0.10000000149011612}}
Ї


Hkernel
Ibias
#J_self_saveable_object_factories
Kregularization_losses
L	variables
Mtrainable_variables
N	keras_api
+љ&call_and_return_all_conditional_losses
Њ__call__"о
_tf_keras_layer‘{"class_name": "Conv2DTranspose", "name": "deconv1", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "deconv1", "trainable": false, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 4}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 4]}}
а
Oregularization_losses
P	variables
Qtrainable_variables
R	keras_api
+њ&call_and_return_all_conditional_losses
ј__call__"ѕ
_tf_keras_layerµ{"class_name": "Activation", "name": "deconv1-sigmoid", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "deconv1-sigmoid", "trainable": true, "dtype": "float32", "activation": "sigmoid"}}
£
Siter

Tbeta_1

Ubeta_2
	Vdecay
Wlearning_rate)mЮ*mЯ3m†4m°)vҐ*v£3v§4v•"
	optimizer
 "
trackable_list_wrapper
v
0
1
2
3
)4
*5
36
47
=8
>9
H10
I11"
trackable_list_wrapper
<
)0
*1
32
43"
trackable_list_wrapper
ќ
Xlayer_metrics
Ymetrics
regularization_losses
	variables
Zlayer_regularization_losses

[layers
\non_trainable_variables
trainable_variables
®__call__
¶_default_save_signature
+І&call_and_return_all_conditional_losses
'І"call_and_return_conditional_losses"
_generic_user_object
-
Ѕserving_default"
signature_map
&:$2conv1/kernel
:2
conv1/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
∞
]layer_metrics
^metrics
regularization_losses
_layer_regularization_losses
	variables

`layers
anon_trainable_variables
trainable_variables
™__call__
+©&call_and_return_all_conditional_losses
'©"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
∞
blayer_metrics
cmetrics
regularization_losses
dlayer_regularization_losses
	variables

elayers
fnon_trainable_variables
trainable_variables
ђ__call__
+Ђ&call_and_return_all_conditional_losses
'Ђ"call_and_return_conditional_losses"
_generic_user_object
&:$
2conv2/kernel
:
2
conv2/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
∞
glayer_metrics
hmetrics
!regularization_losses
ilayer_regularization_losses
"	variables

jlayers
knon_trainable_variables
#trainable_variables
Ѓ__call__
+≠&call_and_return_all_conditional_losses
'≠"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
∞
llayer_metrics
mmetrics
%regularization_losses
nlayer_regularization_losses
&	variables

olayers
pnon_trainable_variables
'trainable_variables
∞__call__
+ѓ&call_and_return_all_conditional_losses
'ѓ"call_and_return_conditional_losses"
_generic_user_object
&:$
2conv3/kernel
:2
conv3/bias
 "
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
∞
qlayer_metrics
rmetrics
+regularization_losses
slayer_regularization_losses
,	variables

tlayers
unon_trainable_variables
-trainable_variables
≤__call__
+±&call_and_return_all_conditional_losses
'±"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
∞
vlayer_metrics
wmetrics
/regularization_losses
xlayer_regularization_losses
0	variables

ylayers
znon_trainable_variables
1trainable_variables
і__call__
+≥&call_and_return_all_conditional_losses
'≥"call_and_return_conditional_losses"
_generic_user_object
(:&
2deconv3/kernel
:
2deconv3/bias
 "
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
∞
{layer_metrics
|metrics
5regularization_losses
}layer_regularization_losses
6	variables

~layers
non_trainable_variables
7trainable_variables
ґ__call__
+µ&call_and_return_all_conditional_losses
'µ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Аlayer_metrics
Бmetrics
9regularization_losses
 Вlayer_regularization_losses
:	variables
Гlayers
Дnon_trainable_variables
;trainable_variables
Є__call__
+Ј&call_and_return_all_conditional_losses
'Ј"call_and_return_conditional_losses"
_generic_user_object
(:&
2deconv2/kernel
:2deconv2/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Еlayer_metrics
Жmetrics
@regularization_losses
 Зlayer_regularization_losses
A	variables
Иlayers
Йnon_trainable_variables
Btrainable_variables
Ї__call__
+є&call_and_return_all_conditional_losses
'є"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Кlayer_metrics
Лmetrics
Dregularization_losses
 Мlayer_regularization_losses
E	variables
Нlayers
Оnon_trainable_variables
Ftrainable_variables
Љ__call__
+ї&call_and_return_all_conditional_losses
'ї"call_and_return_conditional_losses"
_generic_user_object
(:&2deconv1/kernel
:2deconv1/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
H0
I1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Пlayer_metrics
Рmetrics
Kregularization_losses
 Сlayer_regularization_losses
L	variables
Тlayers
Уnon_trainable_variables
Mtrainable_variables
Њ__call__
+љ&call_and_return_all_conditional_losses
'љ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Фlayer_metrics
Хmetrics
Oregularization_losses
 Цlayer_regularization_losses
P	variables
Чlayers
Шnon_trainable_variables
Qtrainable_variables
ј__call__
+њ&call_and_return_all_conditional_losses
'њ"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_dict_wrapper
(
Щ0"
trackable_list_wrapper
 "
trackable_list_wrapper
v
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
11"
trackable_list_wrapper
X
0
1
2
3
=4
>5
H6
I7"
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
0
1"
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
0
1"
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
.
=0
>1"
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
H0
I1"
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
њ

Ъtotal

Ыcount
Ь	variables
Э	keras_api"Д
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
:  (2total
:  (2count
0
Ъ0
Ы1"
trackable_list_wrapper
.
Ь	variables"
_generic_user_object
+:)
2Adam/conv3/kernel/m
:2Adam/conv3/bias/m
-:+
2Adam/deconv3/kernel/m
:
2Adam/deconv3/bias/m
+:)
2Adam/conv3/kernel/v
:2Adam/conv3/bias/v
-:+
2Adam/deconv3/kernel/v
:
2Adam/deconv3/bias/v
ж2г
 __inference__wrapped_model_40734Њ
Л≤З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *.Ґ+
)К&
input_2€€€€€€€€€@@
т2п
I__inference_autoencoder-v3_layer_call_and_return_conditional_losses_41054
I__inference_autoencoder-v3_layer_call_and_return_conditional_losses_41442
I__inference_autoencoder-v3_layer_call_and_return_conditional_losses_41094
I__inference_autoencoder-v3_layer_call_and_return_conditional_losses_41357ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ж2Г
.__inference_autoencoder-v3_layer_call_fn_41233
.__inference_autoencoder-v3_layer_call_fn_41471
.__inference_autoencoder-v3_layer_call_fn_41500
.__inference_autoencoder-v3_layer_call_fn_41164ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
к2з
@__inference_conv1_layer_call_and_return_conditional_losses_41510Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ѕ2ћ
%__inference_conv1_layer_call_fn_41519Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
р2н
F__inference_conv1-leaky_layer_call_and_return_conditional_losses_41524Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
’2“
+__inference_conv1-leaky_layer_call_fn_41529Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
к2з
@__inference_conv2_layer_call_and_return_conditional_losses_41539Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ѕ2ћ
%__inference_conv2_layer_call_fn_41548Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
р2н
F__inference_conv2-leaky_layer_call_and_return_conditional_losses_41553Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
’2“
+__inference_conv2-leaky_layer_call_fn_41558Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
к2з
@__inference_conv3_layer_call_and_return_conditional_losses_41568Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ѕ2ћ
%__inference_conv3_layer_call_fn_41577Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
р2н
F__inference_conv3-leaky_layer_call_and_return_conditional_losses_41582Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
’2“
+__inference_conv3-leaky_layer_call_fn_41587Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
°2Ю
B__inference_deconv3_layer_call_and_return_conditional_losses_40772„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ж2Г
'__inference_deconv3_layer_call_fn_40782„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€
т2п
H__inference_deconv3-leaky_layer_call_and_return_conditional_losses_41592Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
„2‘
-__inference_deconv3-leaky_layer_call_fn_41597Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
°2Ю
B__inference_deconv2_layer_call_and_return_conditional_losses_40820„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€

Ж2Г
'__inference_deconv2_layer_call_fn_40830„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€

т2п
H__inference_deconv2-leaky_layer_call_and_return_conditional_losses_41602Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
„2‘
-__inference_deconv2-leaky_layer_call_fn_41607Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
°2Ю
B__inference_deconv1_layer_call_and_return_conditional_losses_40868„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ж2Г
'__inference_deconv1_layer_call_fn_40878„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€
ф2с
J__inference_deconv1-sigmoid_layer_call_and_return_conditional_losses_41612Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ў2÷
/__inference_deconv1-sigmoid_layer_call_fn_41617Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
2B0
#__inference_signature_wrapper_41272input_2Є
 __inference__wrapped_model_40734У)*34=>HI8Ґ5
.Ґ+
)К&
input_2€€€€€€€€€@@
™ "I™F
D
deconv1-sigmoid1К.
deconv1-sigmoid€€€€€€€€€@@я
I__inference_autoencoder-v3_layer_call_and_return_conditional_losses_41054С)*34=>HI@Ґ=
6Ґ3
)К&
input_2€€€€€€€€€@@
p

 
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ я
I__inference_autoencoder-v3_layer_call_and_return_conditional_losses_41094С)*34=>HI@Ґ=
6Ґ3
)К&
input_2€€€€€€€€€@@
p 

 
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ Ћ
I__inference_autoencoder-v3_layer_call_and_return_conditional_losses_41357~)*34=>HI?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€@@
p

 
™ "-Ґ*
#К 
0€€€€€€€€€@@
Ъ Ћ
I__inference_autoencoder-v3_layer_call_and_return_conditional_losses_41442~)*34=>HI?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€@@
p 

 
™ "-Ґ*
#К 
0€€€€€€€€€@@
Ъ Ј
.__inference_autoencoder-v3_layer_call_fn_41164Д)*34=>HI@Ґ=
6Ґ3
)К&
input_2€€€€€€€€€@@
p

 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€Ј
.__inference_autoencoder-v3_layer_call_fn_41233Д)*34=>HI@Ґ=
6Ґ3
)К&
input_2€€€€€€€€€@@
p 

 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€ґ
.__inference_autoencoder-v3_layer_call_fn_41471Г)*34=>HI?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€@@
p

 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€ґ
.__inference_autoencoder-v3_layer_call_fn_41500Г)*34=>HI?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€@@
p 

 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€≤
F__inference_conv1-leaky_layer_call_and_return_conditional_losses_41524h7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€  
™ "-Ґ*
#К 
0€€€€€€€€€  
Ъ К
+__inference_conv1-leaky_layer_call_fn_41529[7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€  
™ " К€€€€€€€€€  ∞
@__inference_conv1_layer_call_and_return_conditional_losses_41510l7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€@@
™ "-Ґ*
#К 
0€€€€€€€€€  
Ъ И
%__inference_conv1_layer_call_fn_41519_7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€@@
™ " К€€€€€€€€€  ≤
F__inference_conv2-leaky_layer_call_and_return_conditional_losses_41553h7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€

™ "-Ґ*
#К 
0€€€€€€€€€

Ъ К
+__inference_conv2-leaky_layer_call_fn_41558[7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€

™ " К€€€€€€€€€
∞
@__inference_conv2_layer_call_and_return_conditional_losses_41539l7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€  
™ "-Ґ*
#К 
0€€€€€€€€€

Ъ И
%__inference_conv2_layer_call_fn_41548_7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€  
™ " К€€€€€€€€€
≤
F__inference_conv3-leaky_layer_call_and_return_conditional_losses_41582h7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€
™ "-Ґ*
#К 
0€€€€€€€€€
Ъ К
+__inference_conv3-leaky_layer_call_fn_41587[7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€
™ " К€€€€€€€€€∞
@__inference_conv3_layer_call_and_return_conditional_losses_41568l)*7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€

™ "-Ґ*
#К 
0€€€€€€€€€
Ъ И
%__inference_conv3_layer_call_fn_41577_)*7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€

™ " К€€€€€€€€€џ
J__inference_deconv1-sigmoid_layer_call_and_return_conditional_losses_41612МIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ≤
/__inference_deconv1-sigmoid_layer_call_fn_41617IҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€„
B__inference_deconv1_layer_call_and_return_conditional_losses_40868РHIIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ѓ
'__inference_deconv1_layer_call_fn_40878ГHIIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€ў
H__inference_deconv2-leaky_layer_call_and_return_conditional_losses_41602МIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ∞
-__inference_deconv2-leaky_layer_call_fn_41607IҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€„
B__inference_deconv2_layer_call_and_return_conditional_losses_40820Р=>IҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€

™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ѓ
'__inference_deconv2_layer_call_fn_40830Г=>IҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€

™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€ў
H__inference_deconv3-leaky_layer_call_and_return_conditional_losses_41592МIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€

™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€

Ъ ∞
-__inference_deconv3-leaky_layer_call_fn_41597IҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€

™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€
„
B__inference_deconv3_layer_call_and_return_conditional_losses_40772Р34IҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€

Ъ ѓ
'__inference_deconv3_layer_call_fn_40782Г34IҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€
∆
#__inference_signature_wrapper_41272Ю)*34=>HICҐ@
Ґ 
9™6
4
input_2)К&
input_2€€€€€€€€€@@"I™F
D
deconv1-sigmoid1К.
deconv1-sigmoid€€€€€€€€€@@
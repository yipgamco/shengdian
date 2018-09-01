# -*- coding: UTF-8 -*-
import math
import os
import time
from skimage import io, transform
import cv2
import numpy as np
import tensorflow as tf
from config import FLAGS
from utils import tracking_module, utils
from models.nets import cpm_hand

dict = {0:"花",1:"五",2:"六",3:"中指",4:"空白",5:"加油",6:"好运",7:"一",8:"V",9:"okay"}
#  手势字典

joint_detections = np.zeros(shape=(21, 2))	# 创建一个21×2的列表，用于侦查关节

w = 100			# 识别后出来的黑白图的长宽尺寸
h = 100
c = 3
num_of_photo = 50		# 读图数量
num_of_gesture = 9		# 手势数量

do_again = 0
again_time = 0
receive_flag = 0


local_dir = '/home/jim/PycharmProjects/behind'

def main(argv):
    tracker = tracking_module.SelfTracker([FLAGS.webcam_height, FLAGS.webcam_width], FLAGS.input_size)
    model = cpm_hand.CPM_Model(input_size=FLAGS.input_size,         # 导入model1的模型结构
                               heatmap_size=FLAGS.heatmap_size,
                               stages=FLAGS.cpm_stages,
                               joints=FLAGS.num_of_joints,
                               img_type=FLAGS.color_channel,
                               is_training=False)
    print("成功导入1的模型")
    saver = tf.train.Saver()
    output_node = tf.get_default_graph().get_tensor_by_name(name=FLAGS.output_node_names)
    with tf.Session() as sess:
        saver.restore(sess, './models/weights/cpm_hand')        # 恢复model1的参数
        print("成功导入1的model参数")

        tf.train.import_meta_graph('./classify/modelSave/model.ckpt.meta')      # 导入model3的模型结构
        print("成功导入3的model结构")
        all_vars = tf.trainable_variables()
        # print(all_vars[61:])
        saver_fenlei = tf.train.Saver(all_vars[62:])	# 这个saver1只导入第62个变量及以后变量，是model3的专属变量
        saver_fenlei.restore(sess, tf.train.latest_checkpoint('./classify/modelSave/'))
        print("成功导入3的model参数")
        x = tf.get_default_graph().get_tensor_by_name("x:0")
        logits = tf.get_default_graph().get_tensor_by_name("logits_eval:0")


        # Create kalman filters
        if FLAGS.use_kalman:
            kalman_filter_array = [cv2.KalmanFilter(4, 2) for _ in range(FLAGS.num_of_joints)]
            for _, joint_kalman_filter in enumerate(kalman_filter_array):
                joint_kalman_filter.transitionMatrix = np.array(
                    [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
                    np.float32)
                joint_kalman_filter.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
                joint_kalman_filter.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                                                               np.float32) * FLAGS.kalman_noise
        else:
            kalman_filter_array = None


        while True:
            data = []
            global receive_flag,do_again,again_time

            files = os.listdir(local_dir+'/storePic')
            for f in files:
                if f == 'flag_ok':receive_flag = 1


            if receive_flag or (do_again and again_time<2) :
                os.system('rm ./storePic/flag_ok')
                receive_flag = 0
                do_again = 0

                for i in range(num_of_photo):
                    full_img = cv2.imread("./storePic/test" + str(i) + ".jpg")
                    cv2.waitKey(1)
                    cv2.namedWindow("yuantu")
                    cv2.imshow("yuantu", full_img)
                    cv2.moveWindow("yuantu",0,0)
                    test_img = tracker.tracking_by_joints(full_img, joint_detections=joint_detections)
                    crop_full_scale = tracker.input_crop_ratio
                    test_img_copy = test_img.copy()

                    # White balance
                    test_img_wb = utils.img_white_balance(test_img, 5)
                    test_img_input = normalize_and_centralize_img(test_img_wb
                                                                  )
                    t1 = time.time()
                    stage_heatmap_np = sess.run([output_node],
                                                feed_dict={model.input_images: test_img_input})
                    print('FPS: %.2f' % (1 / (time.time() - t1)))

                    local_img = visualize_result(full_img, stage_heatmap_np, kalman_filter_array, tracker,
                                                 crop_full_scale,
                                                 test_img_copy)
                    cv2.namedWindow("local_image")
                    cv2.namedWindow("global_image")
                    cv2.imshow('local_img', local_img.astype(np.uint8))  # 训练用图
                    cv2.imshow('global_img', full_img.astype(np.uint8))  # 单人大框
                    cv2.moveWindow("local_image", 1000, 0)
                    cv2.moveWindow("global_image", 0, 2000)
                    cv2.imwrite("./blackPic/black" + str(i) + ".jpg", local_img)

                    local_img = io.imread("./blackPic/black" + str(i) + ".jpg")
                    img = transform.resize(local_img, (w, h))
                    data.append(np.asarray(img))
                    print(len(data))
                feed_dict = {x: data}
                classification_result = sess.run(logits, feed_dict)

                # 打印出预测矩阵
                print(classification_result)

                # 打印出预测矩阵每一行最大值的索引
                print(tf.argmax(classification_result, 1).eval())

                # 根据索引通过字典对应花的分类
                output = tf.argmax(classification_result, 1).eval()
                for k in range(len(output)):
                    print("第" + str(k + 1) + "帧预测:" + dict[output[k]])
                print("#########################################")
                output = output.tolist()
                num_of_none = find_most(output)

                if num_of_none>int((num_of_photo/2.0)):
                    again_time +=1
                    do_again = 1
                else:
                    again_time = 0
                    do_again = 0
                cv2.destroyAllWindows()
                print('waiting...')


           


def find_most(array):
    liebiao = []
    for i in range(num_of_gesture+1):
        liebiao.append(array.count(i))
        print(dict[i]+"的概率为:%.2f" %(array.count(i)/1.00/num_of_photo*100)+'%')
    idx = liebiao.index(max(liebiao))
    print('#####################################')
    if idx == 4:
        print("图片质量不佳")

    else:
        per = (max(liebiao)/(num_of_photo-liebiao[4]/1.000))*100
        print("空白的个数为："+str(liebiao[4]))
        print("综合预测结果为："+dict[idx])
        print(dict[idx]+"出现次数：%d" % max(liebiao))
        print("准确率为:"+str(per)+"%")
    return liebiao[4]


def normalize_and_centralize_img(img):
    if FLAGS.color_channel == 'GRAY':
        img = np.dot(img[..., :3], [0.299, 0.587, 0.114]).reshape((FLAGS.input_size, FLAGS.input_size, 1))

    if FLAGS.normalize_img:
        test_img_input = img / 256.0 - 0.5
        test_img_input = np.expand_dims(test_img_input, axis=0)
    else:
        test_img_input = img - 128.0
        test_img_input = np.expand_dims(test_img_input, axis=0)
    return test_img_input


def visualize_result(test_img, stage_heatmap_np, kalman_filter_array, tracker, crop_full_scale, crop_img):
    demo_stage_heatmaps = []

    for stage in range(len(stage_heatmap_np)):
        demo_stage_heatmap = stage_heatmap_np[stage][0, :, :, 0:FLAGS.num_of_joints].reshape(
            (FLAGS.heatmap_size, FLAGS.heatmap_size, FLAGS.num_of_joints))
        demo_stage_heatmap = cv2.resize(demo_stage_heatmap, (FLAGS.input_size, FLAGS.input_size))
        demo_stage_heatmap = np.amax(demo_stage_heatmap, axis=2)
        demo_stage_heatmap = np.reshape(demo_stage_heatmap, (FLAGS.input_size, FLAGS.input_size, 1))
        demo_stage_heatmap = np.repeat(demo_stage_heatmap, 3, axis=2)
        demo_stage_heatmap *= 255
        demo_stage_heatmaps.append(demo_stage_heatmap)

    last_heatmap = stage_heatmap_np[len(stage_heatmap_np) - 1][0, :, :, 0:FLAGS.num_of_joints].reshape(
        (FLAGS.heatmap_size, FLAGS.heatmap_size, FLAGS.num_of_joints))
    last_heatmap = cv2.resize(last_heatmap, (FLAGS.input_size, FLAGS.input_size))



    correct_and_draw_hand(test_img, last_heatmap, kalman_filter_array, tracker, crop_full_scale, crop_img)

    if len(demo_stage_heatmaps) > 3:
        upper_img = np.concatenate((demo_stage_heatmaps[0], demo_stage_heatmaps[1], demo_stage_heatmaps[2]), axis=1)
        lower_img = np.concatenate(
            (demo_stage_heatmaps[3], demo_stage_heatmaps[len(stage_heatmap_np) - 1], crop_img),
            axis=1)
        demo_img = np.concatenate((upper_img, lower_img), axis=0)
        return demo_img
    else:
        # return np.concatenate((demo_stage_heatmaps[0], demo_stage_heatmaps[len(stage_heatmap_np) - 1], crop_img),
        #                       axis=1)

        return demo_stage_heatmaps[0]
        # np.concatenate 合并array


def correct_and_draw_hand(full_img, stage_heatmap_np, kalman_filter_array, tracker, crop_full_scale, crop_img):
    global joint_detections
    joint_coord_set = np.zeros((FLAGS.num_of_joints, 2))
    local_joint_coord_set = np.zeros((FLAGS.num_of_joints, 2))

    mean_response_val = 0.0

    # Plot joint colors
    if kalman_filter_array is not None:
        for joint_num in range(FLAGS.num_of_joints):
            tmp_heatmap = stage_heatmap_np[:, :, joint_num]
            joint_coord = np.unravel_index(np.argmax(tmp_heatmap),
                                           (FLAGS.input_size, FLAGS.input_size))
            mean_response_val += tmp_heatmap[joint_coord[0], joint_coord[1]]
            joint_coord = np.array(joint_coord).reshape((2, 1)).astype(np.float32)
            kalman_filter_array[joint_num].correct(joint_coord)
            kalman_pred = kalman_filter_array[joint_num].predict()
            correct_coord = np.array([kalman_pred[0], kalman_pred[1]]).reshape((2))
            local_joint_coord_set[joint_num, :] = correct_coord

            # Resize back
            correct_coord /= crop_full_scale

            # Substract padding border
            correct_coord[0] -= (tracker.pad_boundary[0] / crop_full_scale)
            correct_coord[1] -= (tracker.pad_boundary[2] / crop_full_scale)
            correct_coord[0] += tracker.bbox[0]
            correct_coord[1] += tracker.bbox[2]
            joint_coord_set[joint_num, :] = correct_coord

    else:
        for joint_num in range(FLAGS.num_of_joints):
            tmp_heatmap = stage_heatmap_np[:, :, joint_num]
            joint_coord = np.unravel_index(np.argmax(tmp_heatmap),
                                           (FLAGS.input_size, FLAGS.input_size))
            mean_response_val += tmp_heatmap[joint_coord[0], joint_coord[1]]
            joint_coord = np.array(joint_coord).astype(np.float32)

            local_joint_coord_set[joint_num, :] = joint_coord

            # Resize back
            joint_coord /= crop_full_scale

            # Substract padding border
            joint_coord[0] -= (tracker.pad_boundary[2] / crop_full_scale)
            joint_coord[1] -= (tracker.pad_boundary[0] / crop_full_scale)
            joint_coord[0] += tracker.bbox[0]
            joint_coord[1] += tracker.bbox[2]
            joint_coord_set[joint_num, :] = joint_coord

    draw_hand(full_img, joint_coord_set, tracker.loss_track)
    draw_hand(crop_img, local_joint_coord_set, tracker.loss_track)
    joint_detections = joint_coord_set

    if mean_response_val >= 1:
        tracker.loss_track = False
    else:
        tracker.loss_track = True

    cv2.putText(full_img, 'Response: {:<.3f}'.format(mean_response_val),
                org=(20, 20), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(255, 0, 0))


def draw_hand(full_img, joint_coords, is_loss_track):
    if is_loss_track:
        joint_coords = FLAGS.default_hand

    # Plot joints
    for joint_num in range(FLAGS.num_of_joints):
        color_code_num = (joint_num // 4)
        if joint_num in [0, 4, 8, 12, 16]:
            joint_color = list(map(lambda x: x + 35 * (joint_num % 4), FLAGS.joint_color_code[color_code_num]))
            cv2.circle(full_img, center=(int(joint_coords[joint_num][1]), int(joint_coords[joint_num][0])), radius=3,
                       color=joint_color, thickness=-1)
        else:
            joint_color = list(map(lambda x: x + 35 * (joint_num % 4), FLAGS.joint_color_code[color_code_num]))
            cv2.circle(full_img, center=(int(joint_coords[joint_num][1]), int(joint_coords[joint_num][0])), radius=3,
                       color=joint_color, thickness=-1)

    # Plot limbs
    for limb_num in range(len(FLAGS.limbs)):
        x1 = int(joint_coords[int(FLAGS.limbs[limb_num][0])][0])
        y1 = int(joint_coords[int(FLAGS.limbs[limb_num][0])][1])
        x2 = int(joint_coords[int(FLAGS.limbs[limb_num][1])][0])
        y2 = int(joint_coords[int(FLAGS.limbs[limb_num][1])][1])
        length = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        if length < 150 and length > 5:
            deg = math.degrees(math.atan2(x1 - x2, y1 - y2))
            polygon = cv2.ellipse2Poly((int((y1 + y2) / 2), int((x1 + x2) / 2)),
                                       (int(length / 2), 3),
                                       int(deg),
                                       0, 360, 1)
            color_code_num = limb_num // 4
            limb_color = list(map(lambda x: x + 35 * (limb_num % 4), FLAGS.joint_color_code[color_code_num]))
            cv2.fillConvexPoly(full_img, polygon, color=limb_color)


if __name__ == '__main__':
    tf.app.run()

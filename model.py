import tensorflow as tf

def order_violations(pr, ch):
    """
    Computes the order violations
    """
    return tf.pow(tf.maximum(tf.constant(0, dtype=tf.float64), pr - ch), 2)

def get_classification_errors(im_emb, cl_emb):
    im2 = tf.expand_dims(im_emb, 0)
    c2 = tf.expand_dims(cl_emb, 1)

    errors = tf.reduce_sum(order_violations(c2, im2), axis=2)

    return errors

def get_classification_loss(im_emb, cl_emb, labels, margin):
    errors = get_classification_errors(im_emb, cl_emb)

    inverted_labels = 1 - labels
    
    positive = errors * tf.transpose(labels)
    positive_sum = tf.reduce_max(positive, axis=0)
    negative = errors * tf.transpose(inverted_labels)

    cost = tf.maximum(tf.constant(0, dtype=tf.float64), margin - negative + positive_sum)
    
    cost = cost * tf.transpose(inverted_labels)

    return tf.reduce_sum(cost)


def get_prediction(im, k, options, only_base_classes=True):
    if only_base_classes:
        class_ids = tf.range(options['num_base_classes'])
    else:
        class_ids = tf.range(options['num_all_classes'])

    im_emb = get_image_embedding(im, options)
    cl_emb = get_hypernym_embedding(class_ids, options)

    errors = get_classification_errors(im_emb, cl_emb)
    _, predictions = tf.nn.top_k(tf.transpose(-errors), k)

    return predictions


def get_classification_errors_all_classes(im, options):
    all_class_ids = tf.range(options['num_all_classes'])
    
    im_emb = get_image_embedding(im, options)
    cl_emb = get_hypernym_embedding(all_class_ids, options)
    
    errors = get_classification_errors(im_emb, cl_emb)

    return tf.transpose(errors)

def get_hypernym_errors(ch_emb, pr_emb):
    errors = tf.reduce_sum(order_violations(pr_emb, ch_emb), axis=1)

    return errors

def get_hypernym_loss(pos_ch_emb, pos_pr_emb, neg_ch_emb, neg_pr_emb, margin):
    pos_errors = get_hypernym_errors(pos_ch_emb, pos_pr_emb)
    neg_errors = get_hypernym_errors(neg_ch_emb, neg_pr_emb)

    cost = pos_errors + tf.maximum(tf.constant(0, dtype=tf.float64), margin - neg_errors)
    
    return tf.reduce_sum(cost)

def get_image_embedding(images, options):
    """
    Return model for image embedding
    """
    with tf.variable_scope("cls", reuse=tf.AUTO_REUSE):
        im_emb = tf.layers.dense(images, options['dim'], name='i_emb')

    if options['abs']:
        im_emb = tf.maximum(tf.constant(0, dtype=tf.float64), im_emb)

    return im_emb



def get_hypernym_embedding(synset_ids, options):
    """
    Return model for hypernym embedding
    """
    h_onehot = tf.one_hot(synset_ids, options['num_all_classes'], dtype=tf.float64)

    with tf.variable_scope("hyp", reuse=tf.AUTO_REUSE):
        h_emb = tf.layers.dense(h_onehot, options['dim'], name='h_emb') 
    
    if options['abs']:
        h_emb = tf.maximum(tf.constant(0, dtype=tf.float64), h_emb)
    
    return h_emb


def get_hypernym_model(pos_ch, pos_pr, neg_ch, neg_pr, options):
    
    margin = tf.constant(options['margin'], dtype=tf.float64)   

    pos_ch_emb = get_hypernym_embedding(pos_ch, options)
    pos_pr_emb = get_hypernym_embedding(pos_pr, options)
    neg_ch_emb = get_hypernym_embedding(neg_ch, options)
    neg_pr_emb = get_hypernym_embedding(neg_pr, options)
    
    loss = get_hypernym_loss(pos_ch_emb, pos_pr_emb, neg_ch_emb, neg_pr_emb, margin)
    
    pos_errors = get_hypernym_errors(pos_ch_emb, pos_pr_emb) 
    neg_errors = get_hypernym_errors(neg_ch_emb, neg_pr_emb)
    
    pos_predictions = tf.less(pos_errors, margin)
    neg_predictions = tf.less(neg_errors, margin)

    pos_corr_count = tf.count_nonzero(pos_predictions)
    neg_corr_count = tf.count_nonzero(neg_predictions)

    pos_acc = pos_corr_count / tf.cast(tf.shape(pos_ch)[0], tf.int64)
    neg_acc = 1 - neg_corr_count / tf.cast(tf.shape(neg_ch)[0], tf.int64)

    acc = (pos_acc + neg_acc) / 2 

    return (acc, pos_acc, neg_acc), (pos_errors, neg_errors), loss

def get_classification_model(im, y, options):
    margin = tf.constant(options['margin'], dtype=tf.float64)   
    
    im_emb = get_image_embedding(im, options)

    all_class_ids = tf.range(options['num_base_classes'])
    cl_emb = get_hypernym_embedding(all_class_ids, options)
    
    onehot_labels = tf.one_hot(y, options['num_base_classes'], dtype=tf.float64)
    
    loss = get_classification_loss(im_emb, cl_emb, onehot_labels, margin)

    errors = get_classification_errors(im_emb, cl_emb)
    
    predictions = tf.argmin(errors, axis=0,output_type=tf.int32)
    
    # acc = tf.metrics.accuracy(y,predictions)
    acc = tf.reduce_mean(tf.cast(tf.equal(y, predictions), tf.float32))

    return acc, errors, loss, predictions

def get_hier_classification_model(im, y, hypernyms_per_class, options):
    margin = tf.constant(options['margin'], dtype=tf.float64)   
    gamma = tf.constant(options['gamma'], dtype=tf.float64)

    im_emb = get_image_embedding(im, options)

    all_class_ids = tf.range(options['num_all_classes'])
    cl_emb = get_hypernym_embedding(all_class_ids, options)
    base_class_ids = tf.range(options['num_base_classes'])
    base_cl_emb = get_hypernym_embedding(base_class_ids, options)
    
    onehot_labels = tf.one_hot(y, options['num_base_classes'], dtype=tf.float64)
    y_hier = tf.gather(hypernyms_per_class, y)

    all_cls_loss = get_classification_loss(im_emb, cl_emb, y_hier, margin)
    base_cls_loss = get_classification_loss(im_emb, base_cl_emb, onehot_labels, margin)

    loss = gamma * all_cls_loss + (1 - gamma) * base_cls_loss

    # errors = get_classification_errors(im_emb, cl_emb)
    
    # predictions = tf.argmin(errors, axis=0,output_type=tf.int32)
    
    # acc = tf.metrics.accuracy(y,predictions)
    # acc = tf.reduce_mean(tf.cast(tf.equal(y, predictions), tf.float32))

    base_errors = get_classification_errors(im_emb, base_cl_emb)
    
    base_predictions = tf.argmin(base_errors, axis=0,output_type=tf.int32)
    
    # acc = tf.metrics.accuracy(y,predictions)
    base_acc = tf.reduce_mean(tf.cast(tf.equal(y, base_predictions), tf.float32))

    return base_acc, base_errors, loss, base_predictions

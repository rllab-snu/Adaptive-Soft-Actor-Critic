import numpy as np
import tensorflow as tf
import gym
import time
from spinup.algos.tac import core
from spinup.algos.tac.core import get_vars
from spinup.utils.logx import EpochLogger
class Alpha:
    def __init__(self, alpha_start=0.2, alpha_end=1e-2, max_iter=200, speed=0.1, schedule='constant', gamma=0.99, threshhold=1e-3):
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.schedule = schedule
        self.max_iter = max_iter
        self.speed = speed
        self.count = 0
        self.alpha = alpha_start
        self.gamma = gamma
        self.temp = []
        self.threshold = threshhold
    def __call__(self, ret=None):
        if self.schedule == 'constant':
            return self.alpha_start
        if self.schedule == 'trao':
            return self.alpha
    def gaussian_liklihood(self, x, mu, std):
        return -0.5*(((x-mu)/(std+1e-8))**2 + 2*np.log(std) + np.log(2*np.pi))
    def update_alpha(self, Q, V, log_pi):
        if self.schedule == 'trao':
            self.count += 1
            if self.alpha_end < self.alpha:
                #epsilon = np.mean(np.abs(A))
                A = Q - self.alpha * log_pi - V
                epsilon = np.mean(np.abs(A))
                #square_mean_Q_V = np.mean((Q-V)*(Q-V))
                #mean_Q_V = np.mean((V-Q)*Q)
                #delta_alpha = self.alpha**2 * (1-self.gamma)**2 * mean_Q_V / (4*epsilon*self.gamma*square_mean_Q_V)
                delta_alpha = -self.alpha**2 * (1-self.gamma) / (4*epsilon*self.gamma)

                #p_val = core.gaussian_likelihood(np.mean(Q-V), 0.0, np.sum((Q-V)**2)/(len(Q) - 1))
                #p_val = stats.ttest_1samp(Q-V, popmean=0)[1]
                #if self.count % 1000 == 0 or p_val > 1-np.sqrt(square_mean_Q_V*delta_alpha**2/(2*self.alpha**4)):
                #    print('{} {} {}'.format(p_val, 1-np.sqrt(square_mean_Q_V*delta_alpha**2/(2*self.alpha**4)), p_val > 1-np.sqrt(square_mean_Q_V*delta_alpha**2/(2*self.alpha**4))))
                #if delta_alpha < 0 and p_val > 1-np.sqrt(square_mean_Q_V*delta_alpha**2/(2*self.alpha**4)):
                #    self.alpha += delta_alpha
                S_A = np.square(np.sum(np.square(A))/(len(A) - 1))
                if delta_alpha < 0 and np.abs(np.mean(A))/S_A < self.threshold:
                    self.alpha += delta_alpha
                    return True
                else:
                    return False
                #self.temp.append(np.abs(np.mean(Q-V))/(np.sum((Q-V)**2)/(len(Q) - 1)) < 2.5 * np.sqrt(square_mean_Q_V*delta_alpha**2/(2*self.alpha**4)))
                #if self.count % 1000 == 0:
                    #print(np.mean(self.temp))
                    #self.temp = []
                #if delta_alpha < 0 and np.abs(np.mean(Q-V)) < 0.01*np.sqrt(square_mean_Q_V):
                #    self.alpha += delta_alpha
class EntropicIndex:
    def __init__(self, q_start=1.0, q_end=2.0, max_iter=200, schedule='constant'):
        self.q_start = q_start
        self.q_end = q_end
        self.schedule = schedule
        self.max_iter = max_iter
        self.count = 0
        
    def __call__(self, ret=None):
        if self.schedule == 'constant':
            return self.q_end

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for TAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def reset(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

def tac(env_fn, actor_critic=core.mlp_q_actor_critic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=5000, epochs=200, replay_size=int(1e6), gamma=0.99, 
        polyak=0.995, lr=1e-3,
        alpha=0.2, alpha_schedule='constant',
        q=1.0, q_schedule='constant',
        pdf_type='gaussian',log_type='q-log',
        batch_size=100, start_steps=10000, 
        max_ep_len=1000, logger_kwargs=dict(), save_freq=1,
        steps_per_alpha_update=1, threshold=1e-3):

    alpha = Alpha(alpha_start=alpha,schedule=alpha_schedule, threshhold=threshold)
    entropic_index = EntropicIndex(q_end=q,schedule=q_schedule)
    
    alpha_t = alpha()
    q_t = entropic_index()
    
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    tf.set_random_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Share information about action space with policy architecture
    ac_kwargs['action_space'] = env.action_space
    ac_kwargs['pdf_type'] = pdf_type
    ac_kwargs['log_type'] = log_type

    # Inputs to computation graph
    x_ph, a_ph, x2_ph, r_ph, d_ph = core.placeholders(obs_dim, act_dim, obs_dim, None, None)
    # Coefficient for Tsallis entropy
    alpha_ph = core.scale_holder()
    # Place holder for entropic index
    q_ph = core.entropic_index_holder()
    
    # Main outputs from computation graph
    with tf.variable_scope('main'):
        mu, pi, logp_pi, q1, q2, q1_pi, q2_pi, v = actor_critic(x_ph, a_ph, q_ph, **ac_kwargs)
    
    # Target value network
    with tf.variable_scope('target'):
        _, _, _, _, _, _, _, v_targ = actor_critic(x2_ph, a_ph, q_ph, **ac_kwargs)

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)
    alpha_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=max_ep_len)
    # Count variables
    var_counts = tuple(core.count_vars(scope) for scope in 
                       ['main/pi', 'main/q1', 'main/q2', 'main/v', 'main'])
    print(('\nNumber of parameters: \t pi: %d, \t' + \
           'q1: %d, \t q2: %d, \t v: %d, \t total: %d\n')%var_counts)

    # Min Double-Q:
    min_q_pi = tf.minimum(q1_pi, q2_pi)
    min_q = tf.minimum(q1, q2)

    # Targets for Q and V regression
    q_backup = tf.stop_gradient(r_ph + gamma*(1-d_ph)*v_targ)
    v_backup = tf.stop_gradient(min_q_pi - alpha_ph * logp_pi)

    td_error = tf.stop_gradient(v_backup - v)

    # Soft actor-critic losses
    pi_loss = tf.reduce_mean(alpha_ph * logp_pi - q1_pi)
    q1_loss = 0.5 * tf.reduce_mean((q_backup - q1)**2)
    q2_loss = 0.5 * tf.reduce_mean((q_backup - q2)**2)
    v_loss = 0.5 * tf.reduce_mean((v_backup - v)**2)
    value_loss = q1_loss + q2_loss + v_loss

    # Policy train op 
    # (has to be separate from value train op, because q1_pi appears in pi_loss)
    pi_optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    train_pi_op = pi_optimizer.minimize(pi_loss, var_list=get_vars('main/pi'))

    # Value train op
    # (control dep of train_pi_op because sess.run otherwise evaluates in nondeterministic order)
    value_optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    value_params = get_vars('main/q') + get_vars('main/v')
    with tf.control_dependencies([train_pi_op]):
        train_value_op = value_optimizer.minimize(value_loss, var_list=value_params)

    # Polyak averaging for target variables
    # (control flow because sess.run otherwise evaluates in nondeterministic order)
    with tf.control_dependencies([train_value_op]):
        target_update = tf.group([tf.assign(v_targ, polyak*v_targ + (1-polyak)*v_main)
                                  for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

    # All ops to call during one training step
    step_ops = [pi_loss, q1_loss, q2_loss, v_loss, q1, q2, v, logp_pi, 
                train_pi_op, train_value_op, target_update]

    # Initializing targets to match main variables
    target_init = tf.group([tf.assign(v_targ, v_main)
                              for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    sess.run(target_init)

    # Setup model saving
    logger.setup_tf_saver(sess, inputs={'x': x_ph, 'a': a_ph, 'q' : q_ph, 'alpha' : alpha_ph}, 
                                outputs={'mu': mu, 'pi': pi, 'q1': q1, 'q2': q2, 'v': v})

    def get_action(o, deterministic=False):
        act_op = mu if deterministic else pi
        return sess.run(act_op, feed_dict={x_ph: o.reshape(1,-1), q_ph: q_t})

    def test_agent(n=10):
        global sess, mu, pi, q1, q2, q1_pi, q2_pi
        for j in range(n):
            o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time 
                a = get_action(o, True)   
                o, r, d, _ = test_env.step(a)
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    start_time = time.time()
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    total_steps = steps_per_epoch * epochs

    # Main loop: collect experience in env and update/log each epoch

    count = 0
    update_rate = []
    for t in range(total_steps):

        """
        Until start_steps have elapsed, randomly sample actions
        from a uniform distribution for better exploration. Afterwards, 
        use the learned policy. 
        """
        if t > start_steps:
            a = get_action(o)
        else:
            a = env.action_space.sample()
        
        # Step the env
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len==max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)
        alpha_buffer.store(o, a, r, o2, d)
        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2

        if d or (ep_len == max_ep_len):
            for j in range(ep_len):
                batch = replay_buffer.sample_batch(batch_size)
                feed_dict = {x_ph: batch['obs1'],
                             x2_ph: batch['obs2'],
                             a_ph: batch['acts'],
                             r_ph: batch['rews'],
                             d_ph: batch['done'],
                             alpha_ph: alpha_t,
                             q_ph: q_t
                            }
                outs = sess.run(step_ops, feed_dict)
                logger.store(LossPi=outs[0], LossQ1=outs[1], LossQ2=outs[2],
                             LossV=outs[3], Q1Vals=outs[4], Q2Vals=outs[5],
                             VVals=outs[6], LogPi=outs[7])
                count += 1
                if np.isnan(outs[0]) or np.isnan(outs[7]).any():
                    print(outs)
                    return

                if t > start_steps and alpha.schedule == 'trao':
                    batch = alpha_buffer.sample_batch(alpha_buffer.size)
                    feed_dict = {x_ph: batch['obs1'],
                                 x2_ph: batch['obs2'],
                                 a_ph: batch['acts'],
                                 r_ph: batch['rews'],
                                 d_ph: batch['done'],
                                 alpha_ph: alpha_t,
                                 q_ph: q_t}
                    values = sess.run([min_q, v, logp_pi], feed_dict)
                    q_old = values[0]
                    V = values[1]
                    log_pi = values[2]
                    update_rate.append(alpha.update_alpha(Q=q_old, log_pi=log_pi, V=V))
                    # alpha_buffer.reset(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)
                    alpha_t = alpha()

            '''
            if t > start_steps and alpha.schedule == 'trao':
                batch = alpha_buffer.sample_batch(alpha_buffer.size)
                feed_dict = {x_ph: batch['obs1'],
                             x2_ph: batch['obs2'],
                             a_ph: batch['acts'],
                             r_ph: batch['rews'],
                             d_ph: batch['done'],
                             alpha_ph: alpha_t,
                             q_ph: q_t}
                values = sess.run([min_q, v, td_error], feed_dict)
                q_old = values[0]
                V = values[1]
                A = values[2]
                alpha.update_alpha(Q=q_old, A=A, V=V)
                # alpha_buffer.reset(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)
                alpha_t = alpha()
            '''
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            alpha_buffer.reset(obs_dim=obs_dim, act_dim=act_dim, size=max_ep_len)
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            
        # End of epoch wrap-up
        if t > 0 and t % steps_per_epoch == 0:
            epoch = t // steps_per_epoch

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs-1):
                logger.save_state({'env': env}, None)

            # Test the performance of the deterministic version of the agent.
            test_agent()

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EntIndex', q_t)
            logger.log_tabular('EntCoeff', alpha_t)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('Q1Vals', with_min_and_max=True) 
            logger.log_tabular('Q2Vals', with_min_and_max=True) 
            logger.log_tabular('VVals', with_min_and_max=True) 
            logger.log_tabular('LogPi', with_min_and_max=True)
            logger.log_tabular('LossQ1', average_only=True)
            logger.log_tabular('LossQ2', average_only=True)
            logger.log_tabular('LossV', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()
            
            # Update alpha and q value
            #alpha_t = alpha()
            q_t = entropic_index()
            if update_rate:
                print(np.mean(update_rate))
            update_rate = []

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=300)
    parser.add_argument('--l', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--exp_name', type=str, default='tac')
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--alpha_schedule', type=str, default='constant')
    parser.add_argument('--steps_per_alpha_update', type=int, default=1)
    parser.add_argument('--threshold', type=float, default=1e-3)
    parser.add_argument('--q', type=float, default=1.0)
    parser.add_argument('--q_schedule', type=str, default='constant')
    parser.add_argument('--pdf_type', type=str, default='gaussian')
    parser.add_argument('--log_type', type=str, default='q-log')
    args = parser.parse_args()
    
    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
    tac(lambda : gym.make(args.env), actor_critic=core.mlp_q_actor_critic, #actor_critic=core.mlp_actor_critic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l),
        alpha=args.alpha, alpha_schedule=args.alpha_schedule, q=args.q, q_schedule=args.q_schedule,
        pdf_type=args.pdf_type,log_type=args.log_type,
        gamma=args.gamma, seed=args.seed, epochs=args.epochs,
        logger_kwargs=logger_kwargs,
        steps_per_alpha_update=args.steps_per_alpha_update,
        threshold=args.threshold)

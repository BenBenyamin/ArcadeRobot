## Spring Quarter

- Add paddle-to-ball distance to the observation vector  
- Verify that latent vectors vary across the entire dataset  
- Train an agent to break even so the dataset is not biased (scores were ~20 for AI, 0 for agent) ✅ 
- Inject random noise into the VAE decoder and observe outputs  
- Visualize the dataset’s average latent vector  
- Introduce dropout  

## 9/16

- Model the delay as a buffer, not as NO-OP. The buffer should start with NO-OP and queue actions as it goes.✅ 
- Model observation delay.

## 9/25 - Update

### Overview

**What we have now:**

[Emulated Game] -> [Gym Enviorment] -> [Delay Wrapper] -> [N Frames stacked as observation] -> [PPO] -> [Action]

**Progress Update**

The VAE approach is set aside for now. I was not able to achieve a decent result in recrating the game frames.
The delay wrapper is there to simulate the real life movement delay , which is expected to be around 20 frames (60 FPS) or 10 frames (30 FPS), per last quarters measurement investigation.

At first, I implented the delay as a buffer (using a queue), in which the agent has to clear the buffer first and then do the action:
```python
self.action_queue = deque(
    [NO_OP] * delay_steps, 
    maxlen=delay_steps
)
```
And then:

```python
delayed_action = self.action_queue.popleft()
obs, reward, terminated, truncated, info = self.env.step(delayed_action)
self.action_queue.append(action)
```

It looks kind of odd because it allowed consequitve up and down motions which are not realistic, as seen below:

<video src = "https://github.com/user-attachments/assets/88aa5277-584a-426f-9f1a-6f123c458b8f"></video>

The next approach that I tried was an "inertia" approach. If the agent wants to do an action, it has to for the first N//2 frames do the previous action, N//2 doing NO OP and then doing the actual action. I do this by incrementing the enviorment before letting the model dictate the next action.

```python

```
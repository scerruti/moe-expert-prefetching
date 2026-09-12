# Email Thread: Taylor's Feedback on Project Direction

**Date:** September 11, 2026  
**Thread:** Project design review and model selection  
**Participants:** Taylor Berg-Kirkpatrick, Stephen Cerruti, Waleed Alghaithi, Dvora Celniker

---

## Thread (Chronological)

### 1. Waleed's Initial Apology (3:08 PM)

**From:** Waleed Alghaithi  
**To:** Team

Hello everyone,

I apologize for not attending the last two meetings. Things have been hectic on my side. I'll try to catch up and review the roadmap Stephen has shared.

Best regards,
-Waleed

---

### 2. Stephen's Initial Email to Taylor (3:49 PM) – *Summarized*

**From:** Stephen Cerruti  
**To:** Taylor Berg-Kirkpatrick

Hi Taylor,

Following up on the task you assigned. Dvora and I want to ensure we're aligned on the project direction before we begin building.

As we understand it, the goal is to investigate whether we can predict expert activations during prefill and use that for speculative weight prefetching to reduce GPU pipeline stalls. We plan to start with Mixtral 8x7B, profile routing patterns on GSM8K and MBPP to evaluate deterministic and domain-specific routing, and then train a predictor to validate the approach.

I've outlined a five-phase roadmap in a design document and organized our research on GitHub: https://github.com/scerruti/moe-expert-prefetching

Before moving into Phase 1, please let us know your thoughts on the following:

- Dataset scope: GSM8K (full 8.5k vs. narrowed arithmetic subset) and MBPP (sanitized 427 vs. full 974)
- Model/Platform: Confirming Mixtral 8x7B on RunPod for the MVP

Looking forward to your feedback.

Best,
Stephen

---

### 3. Taylor's Response – Model Suggestion (3:23 PM)

**From:** Taylor Berg-Kirkpatrick  
**To:** Stephen, Waleed

Hi all!

@Waleed: No worries! Hope things are calming down for you!

@Stephen: Goddamn!!! You guys nailed it! Exactly what I was suggesting... written very clearly... and repo looks like a good skeleton for getting claude code chugging... the dataset and models seem like a fine choice, though mixtral is a little obscure now and you dont need a vision and language model. 

What about something like this one? https://huggingface.co/Qwen/Qwen3.8-27B

People love it... it's supposed to be really good

---

### 4. Taylor's Follow-up Clarification (3:48 PM)

**From:** Taylor Berg-Kirkpatrick  
**To:** Stephen

it just might not be really any harder to do a 27B mode vs an 8B model if they both fit on runpod gpu... but mixtral also good to try!

---

### 5. Taylor's Second Follow-up (3:48 PM)

**From:** Taylor Berg-Kirkpatrick  
**To:** Stephen

oh actually that Qwen model might not even be MoE, so nvm!

---

### 6. Stephen's Question on Model Approach (3:48 PM)

**From:** Stephen Cerruti  
**To:** Taylor

The one question we had about the model selection was if we could use something small like Mixtral just to prove the code works (possibly on Colab) and then move testing to a real model on runpod. Do you think that will work or will it just be more effort in the long run?

---

### 7. Waleed Requests Push Access (10:21 PM)

**From:** Waleed Alghaithi  
**To:** Stephen

Hello Steven,

Can you give me access to pushing on your repository so I can make some minor changes to contributors and styling.

Best regards,
-Waleed

---

### 8. Stephen Grants Access (11:25 PM)

**From:** Stephen Cerruti  
**To:** Waleed

Done, I think. Let me know if I need to change any security settings.

---

### 9. Waleed Confirms (11:37 PM)

**From:** Waleed Alghaithi  
**To:** Stephen

I've accepted it. I'll check tomorrow and make changes if needed.

---

## Summary of Feedback & Action Items

| Item | Status | Action Needed |
|------|--------|---------------|
| **Project direction** | ✅ Approved | None – Taylor loved the design |
| **Dataset selection** | ✅ Approved | None – GSM8K + MBPP approach confirmed |
| **Model selection** | ⚠️ Uncertain | **CLARIFY:** Mixtral vs. alternatives; Taylor suggested Qwen3.8-27B but noted it might not be MoE |
| **Colab → RunPod approach** | ⚠️ Needs input | **CLARIFY:** Is MVP on Colab (Mixtral) then scale to RunPod valid? |
| **Contributors & styling** | 🔄 In progress | Waleed to make changes |

---

## Key Quotes

> "Goddamn!!! You guys nailed it! Exactly what I was suggesting... written very clearly... and repo looks like a good skeleton for getting claude code chugging..."

> "mixtral is a little obscure now and you dont need a vision and language model"

> "it just might not be really any harder to do a 27B mode vs an 8B model if they both fit on runpod gpu... but mixtral also good to try!"

> "oh actually that Qwen model might not even be MoE, so nvm!"

---

**Last Updated:** 2026-09-12  
**Next Step:** Clarify model strategy (Mixtral vs. Qwen alternatives)

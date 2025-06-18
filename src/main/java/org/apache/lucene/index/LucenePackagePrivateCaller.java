/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.apache.lucene.index;

import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.security.AccessController;
import java.security.PrivilegedAction;

public class LucenePackagePrivateCaller {

    public static Object callPrivateFieldWithMethod(Class<?> clz, String fieldName, String methodName, Object called) {
        return AccessController.doPrivileged((PrivilegedAction<Object>) () -> {
            try {
                Field field = clz.getDeclaredField(fieldName);
                field.setAccessible(true);
                return callMethod(field.getDeclaringClass(), methodName, null, called, null);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        });
    }
    public static Object callMethod(Class<?> clz, String methodName, Class<?>[] argTypes, Object called, Object[] args) {
        return AccessController.doPrivileged((PrivilegedAction<Object>) () -> {
            try {
                Method method = clz.getDeclaredMethod(methodName, argTypes);
                return callMethod(method, called, args);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        });
    }

    public static Object callMethod(Method method, Object called, Object... args) {
        return AccessController.doPrivileged((PrivilegedAction<Object>) () -> {
            try {
                method.setAccessible(true);
                return method.invoke(called, args);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        });
    }
}
